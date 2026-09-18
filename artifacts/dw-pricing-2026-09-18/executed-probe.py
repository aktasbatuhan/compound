"""Four cache-enabled calls with isolated billing-meter reconciliation. No retries."""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from compound.cache_policy import mark_cache_prefix

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
RATES = {'priority': (.15, .01, .60), 'flex': (.12, .01, .48)}


def estimate(usage, tier):
    p, o = usage['prompt_tokens'], usage['completion_tokens']
    r = usage.get('cache_read_input_tokens', 0)
    w = usage.get('cache_creation_input_tokens', 0)
    if any(type(v) is not int or v < 0 for v in (p, o, r, w)) or r+w > p:
        raise ValueError('invalid token counters')
    i, c, out = RATES[tier]
    return ((p-r-w)*i + r*c + w*2*i + o*out)/1e6


def meter():
    day = dt.datetime.now(dt.UTC).date()
    result = subprocess.run(['dw', 'usage', '--since', str(day), '--until',
        str(day + dt.timedelta(days=1)), '--output', 'json'],
        capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def model_row(snapshot):
    return next((r for r in snapshot['by_model'] if r['model'] == MODEL),
        dict(request_count=0, input_tokens=0, output_tokens=0, cost=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--go', action='store_true')
    args=parser.parse_args()
    if not args.go:
        print('Dry run: 4 calls, 64 output tokens each, $0.15 reservation cap; explicit 1h cache markers.')
        return
    key=os.environ.get('DOUBLEWORD_API_KEY')
    if not key:
        for line in Path('.env').read_text().splitlines():
            if line.startswith('DOUBLEWORD_API_KEY='):
                key=line.split('=',1)[1].strip().strip('"').strip("'")
    if not key:
        raise ValueError('missing Doubleword key')
    args.out.mkdir(parents=True, exist_ok=False)
    report={'model':MODEL,'cap_usd':.15,'reserved_usd':0,'calls':[],
        'execution':'local diagnostic, not a benchmark run',
        'rate_source':'https://docs.doubleword.ai/inference-api/models/deepseek-ai-deepseek-v4-1-flash',
        'tier_evidence':'requested policy; meter has no tier field'}
    def save():
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        previous=meter()
        time.sleep(12)
        baseline=meter()
        if previous != baseline:
            raise RuntimeError('billing activity during quiet baseline; cannot isolate probe')
        filler=' '.join(f'item-{i:05d} record sequence entry' for i in range(1600))
        for tier in ('priority','flex'):
            prefix=uuid.uuid4().hex+'\n'+filler
            for repeat in range(2):
                payload={'model':MODEL,'messages':mark_cache_prefix([
                    {'role':'user','content':prefix+'\nReply with OK.'}], ttl='1h'),
                    'max_tokens':64,'reasoning_effort':'medium','service_tier':tier}
                wire=json.dumps(payload).encode()
                reserve=(len(wire)+4096)*.30/1e6+64*.60/1e6
                if report['reserved_usd']+reserve > report['cap_usd']:
                    raise RuntimeError('reservation cap reached')
                report['reserved_usd']+=reserve
                row={'requested_tier':tier,'repeat':repeat,'before':baseline,
                    'reservation_usd':reserve,'request_sha256':hashlib.sha256(wire).hexdigest()}
                report['calls'].append(row)
                (args.out/f'request-{len(report["calls"])}.json').write_bytes(wire)
                save()
                start=time.monotonic()
                request=urllib.request.Request('https://api.doubleword.ai/v1/chat/completions',
                    data=wire,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
                with urllib.request.urlopen(request,timeout=180) as response:
                    raw=json.load(response)
                row.update(duration_s=time.monotonic()-start,response=raw)
                save()
                if raw.get('error') or not raw.get('usage'):
                    raise RuntimeError('provider error or missing usage; stop without retry')
                row['derived_usd']=estimate(raw['usage'],tier)
                base=model_row(baseline)
                deadline=time.monotonic()+180
                while True:
                    after=meter()
                    delta=model_row(after)['request_count']-base['request_count']
                    if delta or time.monotonic()>deadline:
                        break
                    time.sleep(10)
                end=model_row(after)
                row['after']=after
                row['meter_delta_usd']=float(end['cost'])-float(base['cost'])
                row['request_delta']=delta
                row['input_delta']=end['input_tokens']-base['input_tokens']
                row['output_delta']=end['output_tokens']-base['output_tokens']
                row['reconciled']= (delta==1 and row['input_delta']==raw['usage']['prompt_tokens']
                    and row['output_delta']==raw['usage']['completion_tokens'])
                save()
                print(json.dumps({k:row[k] for k in ('requested_tier','repeat','derived_usd','meter_delta_usd','reconciled')}),flush=True)
                if not row['reconciled']:
                    raise RuntimeError('meter counts do not match; no further paid calls')
                baseline=after
        report['status']='complete'
        report['billed_usd']=sum(c['meter_delta_usd'] for c in report['calls'])
    except Exception as exc:
        report['status']='stopped'
        report['error_type']=type(exc).__name__
        report['error']=str(exc)[:300]
        raise
    finally:
        save()

if __name__=='__main__':
    main()
