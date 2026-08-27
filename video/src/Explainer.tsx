import React from "react";
import { AbsoluteFill, Sequence, staticFile } from "remotion";
import { Audio } from "@remotion/media";
import { linearTiming, TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { RadarTeaser } from "./scenes/RadarTeaser";
import { Hook } from "./scenes/Hook";
import { Pipeline } from "./scenes/Pipeline";
import { TbRun } from "./scenes/TbRun";
import { Sweep } from "./scenes/Sweep";
import { Frontier } from "./scenes/Frontier";
import { Radar } from "./scenes/Radar";
import { Gate } from "./scenes/Gate";
import { Close } from "./scenes/Close";

// Order (per Codex): radar teaser cold-open -> the 0-55% hook -> how it works ->
// concrete run -> the bar chart -> the frontier -> full radar payoff -> gate ->
// close. Providers table cut to keep momentum.
// total = sum(scenes) - 15 * (# transitions) = 1685 - 120 = 1565 = ~52s.
const D = { teaser: 90, hook: 120, pipeline: 120, tbrun: 220, sweep: 240, frontier: 185, radar: 360, gate: 190, close: 160 };
export const TOTAL = 1565;

const XFADE = 15;
const t = () => <TransitionSeries.Transition timing={linearTiming({ durationInFrames: XFADE })} presentation={fade()} />;

// SFX one-shots keyed to visual beats (overlap-adjusted global frames).
const sfx: { at: number; file: string; vol: number }[] = [
  { at: 0, file: "sfx/el_whoosh.mp3", vol: 0.45 }, // cold-open punch
  { at: 72, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 177, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 282, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 303, file: "sfx/el_click.mp3", vol: 0.4 }, // tb command
  { at: 487, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 530, file: "sfx/el_click.mp3", vol: 0.4 }, // bars fire
  { at: 712, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 882, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 937, file: "sfx/el_click.mp3", vol: 0.35 }, // radar payoff beats
  { at: 993, file: "sfx/el_click.mp3", vol: 0.35 },
  { at: 1049, file: "sfx/el_click.mp3", vol: 0.35 },
  { at: 1105, file: "sfx/el_click.mp3", vol: 0.35 },
  { at: 1227, file: "sfx/el_whoosh.mp3", vol: 0.5 },
  { at: 1334, file: "sfx/el_chime.mp3", vol: 0.6 }, // verdict
  { at: 1402, file: "sfx/el_whoosh.mp3", vol: 0.5 },
];

export const Explainer: React.FC = () => {
  return (
    <AbsoluteFill>
      <TransitionSeries>
        <TransitionSeries.Sequence durationInFrames={D.teaser}><RadarTeaser /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.hook}><Hook /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.pipeline}><Pipeline /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.tbrun}><TbRun /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.sweep}><Sweep /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.frontier}><Frontier /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.radar}><Radar /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.gate}><Gate /></TransitionSeries.Sequence>
        {t()}
        <TransitionSeries.Sequence durationInFrames={D.close}><Close /></TransitionSeries.Sequence>
      </TransitionSeries>

      <Audio src={staticFile("music/music_el.mp3")} volume={0.85} />
      {sfx.map((s, i) => (
        <Sequence key={i} from={s.at}>
          <Audio src={staticFile(s.file)} volume={s.vol} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
