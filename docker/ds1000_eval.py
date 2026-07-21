"""Execute one DS-1000 test program inside an isolated container."""

import json
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    namespace: dict = {}
    try:
        exec(payload["program"], namespace)
    except BaseException as error:
        print(json.dumps({"passed": False, "feedback": f"{type(error).__name__}: {error}"}))
        return
    print(json.dumps({"passed": True, "feedback": "all executable assertions passed"}))


if __name__ == "__main__":
    main()
