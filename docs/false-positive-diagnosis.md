# Per-image diagnosis of no-screen probe false positives

Task 2b, phase one.

Model: `models/task1_candidates/gatekeeper_task1_C_wide_uniform_int8.tflite`, the task1 winner,
at threshold 0.4. Measured under int8 deployment preprocessing: cv2 greyscale, resize to 96
with INTER_AREA, quantise. Probe: no-screen, 235 images, with 0 removed by the Pexels-ID
leakage check. The probe is used for evaluation only and never enters any training set.

## Overview

- False positives, meaning judged as record at score at or above 0.4: 59/235 = 0.251
- Correct rejections: 176/235
- Score distribution over all images: min 0.000, median 0.160, max 0.922

## By scene

Sorted by false-positive rate.

| Scene (subdirectory) | n | FP | FP rate | Mean score | Mean face count |
|---|---:|---:|---:|---:|---:|
| office_colleagues_conversation | 34 | 19 | 0.559 | 0.457 | 0.41 |
| family_home_living_room | 25 | 13 | 0.52 | 0.383 | 0.08 |
| people_meeting_room_talking | 34 | 11 | 0.324 | 0.333 | 0.29 |
| coworkers_standing_meeting | 29 | 7 | 0.241 | 0.258 | 0.59 |
| group_friends_indoor_candid | 29 | 6 | 0.207 | 0.197 | 0.9 |
| people_street_candid | 25 | 2 | 0.08 | 0.15 | 0.2 |
| friends_cafe_group | 29 | 1 | 0.034 | 0.076 | 0.52 |
| people_restaurant_dining | 30 | 0 | 0.0 | 0.098 | 0.23 |

## False positives against correct rejections

Means across each group.

| Dimension | FP (n=59) | Correct rejection (n=176) | Difference |
|---|---:|---:|---:|
| Brightness | 0.601 | 0.415 | +0.186 |
| Contrast | 0.253 | 0.231 | +0.022 |
| Face-count proxy | 0.339 | 0.432 | −0.093 |
| Screen-like rectangle hit rate | 0.237 | 0.119 | +0.118 |

## Full false-positive list

All 59, by descending score.

| # | score | Scene | File | Faces | Brightness | Screen-like |
|---:|---:|---|---|---:|---:|---:|
| 1 | 0.9219 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0024_4343205.jpeg | 0 | 0.6117 | 0 |
| 2 | 0.9102 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0010_4343207.jpeg | 0 | 0.6157 | 0 |
| 3 | 0.8984 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0030_7432116.jpeg | 0 | 0.7474 | 0 |
| 4 | 0.8828 | family_home_living_room | family_home_living_room/family_home_living_room_0022_8120951.jpeg | 0 | 0.7223 | 1 |
| 5 | 0.8789 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0027_7845080.jpeg | 1 | 0.5424 | 0 |
| 6 | 0.8594 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0006_7653572.jpeg | 0 | 0.7476 | 1 |
| 7 | 0.8477 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0017_6950047.jpeg | 0 | 0.5993 | 0 |
| 8 | 0.8125 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0028_7432114.jpeg | 0 | 0.6712 | 0 |
| 9 | 0.7969 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0012_8204404.jpeg | 0 | 0.7328 | 1 |
| 10 | 0.7891 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0023_7964185.jpeg | 1 | 0.57 | 0 |
| 11 | 0.7812 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0019_6950093.jpeg | 0 | 0.5016 | 1 |
| 12 | 0.7773 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0014_7495168.jpeg | 1 | 0.6464 | 1 |
| 13 | 0.7461 | family_home_living_room | family_home_living_room/family_home_living_room_0006_8120953.jpeg | 0 | 0.6703 | 0 |
| 14 | 0.7266 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0024_7652461.jpeg | 0 | 0.5583 | 1 |
| 15 | 0.7266 | family_home_living_room | family_home_living_room/family_home_living_room_0025_8120623.jpeg | 0 | 0.6591 | 0 |
| 16 | 0.707 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0021_7993566.jpeg | 0 | 0.649 | 0 |
| 17 | 0.6992 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0009_7644014.jpeg | 0 | 0.6335 | 0 |
| 18 | 0.6914 | family_home_living_room | family_home_living_room/family_home_living_room_0002_3875141.jpeg | 0 | 0.6869 | 1 |
| 19 | 0.6836 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0025_4347461.jpeg | 0 | 0.5232 | 1 |
| 20 | 0.6758 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0020_165907.jpeg | 0 | 0.5245 | 1 |
| 21 | 0.668 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0017_8068143.jpeg | 2 | 0.6775 | 0 |
| 22 | 0.668 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0022_7433844.jpeg | 0 | 0.659 | 1 |
| 23 | 0.6602 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0003_8204399.jpeg | 1 | 0.6939 | 0 |
| 24 | 0.6602 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0024_8133989.jpeg | 0 | 0.4259 | 0 |
| 25 | 0.6406 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0015_8068146.jpeg | 1 | 0.6784 | 0 |
| 26 | 0.6406 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0027_6950159.jpeg | 1 | 0.6551 | 0 |
| 27 | 0.6328 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0026_7964354.jpeg | 1 | 0.5067 | 0 |
| 28 | 0.6172 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0002_7964210.jpeg | 0 | 0.4083 | 0 |
| 29 | 0.6094 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0018_8068161.jpeg | 0 | 0.6255 | 0 |
| 30 | 0.5977 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0007_7433850.jpeg | 0 | 0.6488 | 0 |
| 31 | 0.5898 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0001_8602417.png | 0 | 0.7359 | 1 |
| 32 | 0.5586 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0005_4345107.jpeg | 0 | 0.6223 | 0 |
| 33 | 0.5586 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0029_7433859.jpeg | 0 | 0.5546 | 0 |
| 34 | 0.5508 | family_home_living_room | family_home_living_room/family_home_living_room_0014_36777501.jpeg | 0 | 0.5926 | 0 |
| 35 | 0.543 | family_home_living_room | family_home_living_room/family_home_living_room_0019_17158663.jpeg | 0 | 0.6577 | 0 |
| 36 | 0.5352 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0010_7869114.jpeg | 0 | 0.4971 | 0 |
| 37 | 0.5352 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0009_36393928.jpeg | 2 | 0.6402 | 1 |
| 38 | 0.5352 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0030_8847199.jpeg | 1 | 0.3025 | 0 |
| 39 | 0.5234 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0016_14340485.jpeg | 1 | 0.6145 | 0 |
| 40 | 0.5234 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0017_7793921.jpeg | 0 | 0.713 | 0 |
| 41 | 0.5156 | family_home_living_room | family_home_living_room/family_home_living_room_0003_280239.jpeg | 0 | 0.7144 | 0 |
| 42 | 0.5078 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0031_32441078.jpeg | 1 | 0.6973 | 0 |
| 43 | 0.5 | family_home_living_room | family_home_living_room/family_home_living_room_0023_28272350.jpeg | 0 | 0.7674 | 0 |
| 44 | 0.4766 | family_home_living_room | family_home_living_room/family_home_living_room_0018_35430055.jpeg | 0 | 0.5608 | 0 |
| 45 | 0.4648 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0004_7433930.jpeg | 1 | 0.5215 | 0 |
| 46 | 0.457 | family_home_living_room | family_home_living_room/family_home_living_room_0005_34541788.jpeg | 0 | 0.4529 | 0 |
| 47 | 0.457 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0019_7964507.jpeg | 0 | 0.5119 | 1 |
| 48 | 0.4492 | friends_cafe_group | friends_cafe_group/friends_cafe_group_0021_20140970.jpeg | 0 | 0.4479 | 0 |
| 49 | 0.4492 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0003_8367619.jpeg | 0 | 0.6223 | 0 |
| 50 | 0.4414 | family_home_living_room | family_home_living_room/family_home_living_room_0008_7114188.jpeg | 0 | 0.6906 | 0 |
| 51 | 0.4414 | people_meeting_room_talking | people_meeting_room_talking/people_meeting_room_talking_0008_23496874.jpeg | 0 | 0.6013 | 0 |
| 52 | 0.4336 | family_home_living_room | family_home_living_room/family_home_living_room_0007_8763082.jpeg | 1 | 0.5163 | 1 |
| 53 | 0.418 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0005_7876794.jpeg | 0 | 0.6934 | 0 |
| 54 | 0.418 | people_street_candid | people_street_candid/people_street_candid_0025_33259432.jpeg | 0 | 0.5726 | 0 |
| 55 | 0.4102 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0015_6930265.jpeg | 1 | 0.6252 | 0 |
| 56 | 0.4102 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0021_8826744.jpeg | 3 | 0.5651 | 0 |
| 57 | 0.4102 | people_street_candid | people_street_candid/people_street_candid_0016_32242667.jpeg | 0 | 0.288 | 0 |
| 58 | 0.4023 | family_home_living_room | family_home_living_room/family_home_living_room_0013_6957830.jpeg | 0 | 0.6551 | 0 |
| 59 | 0.4023 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0015_10782244.jpeg | 0 | 0.409 | 0 |
## Near-threshold correct rejections

The 9 images scoring between 0.32 and 0.4. These are the borderline cases most likely to flip,
and the ones a little more of the same kind of negative would most plausibly push down.

| score | Scene | File |
|---:|---|---|
| 0.3906 | family_home_living_room | family_home_living_room/family_home_living_room_0017_8583811.jpeg |
| 0.375 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0027_10423482.jpeg |
| 0.3594 | coworkers_standing_meeting | coworkers_standing_meeting/coworkers_standing_meeting_0012_7964369.jpeg |
| 0.3438 | people_street_candid | people_street_candid/people_street_candid_0012_13200581.jpeg |
| 0.3398 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0002_27869785.jpeg |
| 0.3398 | group_friends_indoor_candid | group_friends_indoor_candid/group_friends_indoor_candid_0017_27868438.jpeg |
| 0.3398 | office_colleagues_conversation | office_colleagues_conversation/office_colleagues_conversation_0009_3867842.jpeg |
| 0.3242 | family_home_living_room | family_home_living_room/family_home_living_room_0010_7114420.jpeg |
| 0.3242 | people_restaurant_dining | people_restaurant_dining/people_restaurant_dining_0007_12181619.jpeg |
## What the false positives have in common

A note on measurement first. This diagnosis uses the seed 42 deployment artifact, a single
model, giving FP 0.251, while the task1 report's no-screen FP of 0.331 ± 0.091 is a 5-seed
mean. 0.251 falls inside that interval, so the single model is slightly optimistic. The
*structure* of the errors does not depend on which model is used, so the conclusions below hold
regardless.

### 1. False positives concentrate in indoor office, meeting and home scenes

By scene the rates split into clear clusters:

- High, 0.32 to 0.56: `office_colleagues_conversation` 0.559, `family_home_living_room` 0.52,
  `people_meeting_room_talking` 0.324
- Middle, 0.21 to 0.24: `coworkers_standing_meeting` 0.241, `group_friends_indoor_candid` 0.207
- Zero to low, 0 to 0.08: `people_street_candid` 0.08, `friends_cafe_group` 0.034,
  `people_restaurant_dining` 0.0

The model is not being fooled by people. It is being fooled by the built indoor environments
that the positive class lives in — offices, meeting rooms and living rooms are exactly where
monitors, whiteboards, projectors, televisions and documents are found. Bright walls, windows,
picture frames, switched-off screens and whiteboards, and bookshelves in those rooms carry the
geometry and brightness cues that usually sit next to a screen or a document. The gatekeeper
fires even when there is no readable text anywhere in the frame and a person is present.
Outdoor street scenes and dining, which have food, dim light and no large rectangular
screen-like structure, barely trigger at all.

### 2. False positives are brighter, and twice as likely to contain a screen-like rectangle

Brightness is 0.601 for false positives against 0.415 for correct rejections, the largest
single-dimension gap in the comparison.

Screen-like rectangle hit rate is 0.237 against 0.119, roughly double. Windows, picture frames,
switched-off displays and whiteboards are large bright quadrilaterals and act as a misleading
geometric cue. The detector is a heuristic and noisy, and the relationship only holds at the
group level: about a quarter of the false positives carry this cue, so it is a contributing
factor rather than the sole cause.

### 3. Face count is negatively correlated (−0.093), which is the opposite of the intuition

False positives average 0.34 faces against 0.43 for correct rejections, and several of the
highest-scoring false positives contain zero detected frontal faces.

This rules out the count-imbalance hypothesis again. False triggering is driven by the
environment and the background, not by how many people are in frame. It agrees with the task1
conclusion that the real bottleneck is covariate shift rather than a count imbalance, and it
means scattering another batch of people photographs at the problem will achieve little. What
needs covering is that class of indoor environment itself, together with screen-like surfaces
that carry no text.

### Tension with the task2 design

This is the part that needs a decision.

The people negatives added in task2 deliberately excluded office, meeting room and classroom
scenes. One stated reason was that the probe leans toward office and meeting scenes, so
mirroring those scenes in the training negatives would stop the probe being a fair held-out
generalisation test. That reasoning is recorded in the `_design` field of
`keywords_task2_neg_people.json`.

This diagnosis shows the bet behind that choice did not pay off. The assumption was that
"a person is not a reason to record" would generalise from street and market scenes to offices
and living rooms. In built indoor environments, it did not.

What follows from that:

- The gatekeeper cannot reach these high-confusion indoor scenes by generalisation. They need
  in-distribution coverage.
- The cost is methodological. Once office, meeting room and living room negatives enter
  training, the probe stops being a held-out generalisation test on those scenes and becomes an
  in-distribution one, so part of any FP reduction would be "trained on this kind of image"
  rather than genuine generalisation. This is a trade-off to be decided, not a free win.
- There is a second risk. Office, meeting room and living room images very easily contain a
  readable screen or whiteboard text, which would make them positives. Labelling those as
  negatives would contaminate the negative class. The new keywords therefore lean deliberately
  toward empty, blank, off and textless variants — empty rooms, blank whiteboards, switched-off
  monitors and televisions, empty projector screens — to keep contamination low. Even then,
  manual QC to remove any image containing a readable text screen is required before they enter
  training. That is the established task2 protocol and it is a manual step.
