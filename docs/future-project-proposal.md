# Successor Project Proposal: Multimodal Motion Learning Platform

## Document status and relationship to this repository

This document is a proposal for a **separate successor project**. It does not
change the scope, requirements, results, or completion status of the KSAS
XROCKET academic project in this repository. The existing project is Phase 0:
a reproducible research baseline for studying explainable classification of
American Kenpo Karate movements from smartphone inertial signals.

Nothing described below is implemented here. Terms such as *collector*,
*backend*, *watch application*, *coaching*, and *computer vision* refer to
future work unless explicitly identified as an existing Phase 0 artifact.
The current [`PRD.md`](PRD.md) remains authoritative for this repository.

## Motivation from Phase 0

Phase 0 established a participant-independent modeling and explanation
pipeline, but it also identified measurement limits that a successor study
should address:

- the KSAS recordings do not retain sensor-event timestamps;
- realized sample rate, jitter, and cross-sensor synchronization cannot be
  reconstructed;
- participant-level device placement and orientation are not verified;
- the recordings contain movement labels but not expert assessments of
  correctness, technique quality, or learning progress;
- the model is tied to short, pre-segmented recordings from a smartphone worn
  along the forearm; and
- the padded representation is sensitive to sequence boundaries.

These findings motivate better data acquisition before real-time instruction.
They do not validate automated coaching. In particular, the current model
must not be deployed directly on smartwatch data: moving the sensor from a
forearm-mounted phone to a wristwatch changes the device, placement, coordinate
frame, sampling behavior, and motion distribution. New recordings, validation,
and model training are required.

## Vision and research objective

The long-term vision is a learning system that combines wearable inertial
signals with a phone camera to provide timely, understandable feedback about
physical skills. American Kenpo Karate is the first research domain because it
provides a bounded movement vocabulary and continuity with KSAS. The eventual
ambition is a curriculum-driven platform that can support other sports, but
sport generalization is a research direction rather than an MVP claim.

The immediate objective is narrower: build a controlled, timestamp-aware data
collection system for invited study participants using a paired Android phone
and Wear OS watch. The first system collects evidence; it does not teach,
score, diagnose, or prescribe.

## Collector MVP

### Participants and study operation

The MVP targets invited participants enrolled in researcher-managed studies.
A researcher defines the study protocol and issues an invitation. A
participant reviews the applicable consent information, enrolls under a
pseudonymous research identifier, pairs approved devices, completes placement
and calibration checks, and records prompted movement sessions.

The phone is the session coordinator and upload gateway. It must provide:

- invitation-based enrollment and consent acknowledgement;
- protocol, movement, arm, and repetition instructions;
- phone/watch connectivity, placement, calibration, storage, battery, and
  sensor diagnostics;
- synchronized start, marker, stop, retry, and cancellation controls;
- local session review and explicit upload state; and
- resumable upload of a validated, complete research bundle.

The Wear OS application is a focused capture companion. It must record wrist
IMU events, retain them locally through temporary disconnection, accept session
commands and repetition markers, expose capture status, and transfer completed
chunks to the phone. It is not a miniature copy of the phone interface.

### Technology boundary

The proposed phone application uses Flutter for the Android participant and
research workflow. Android-specific sensor acquisition, foreground execution,
and Wear OS communication sit behind narrow native interfaces so the research
data contract is not coupled to a third-party Flutter sensor package.

The watch companion uses Kotlin and Compose for Wear OS with native Android
sensor APIs. Android recommends Compose for Wear OS interfaces, and the Wearable
Data Layer API is the supported paired-device communication channel. Each
native `SensorEvent` provides an event timestamp in nanoseconds on a monotonic
time base. See the official documentation for
[`SensorEvent`](https://developer.android.com/reference/android/hardware/SensorEvent),
the [Wear OS Data Layer](https://developer.android.com/training/wearables/data/overview),
and [Compose for Wear OS](https://developer.android.com/training/wearables/compose).

The Data Layer carries session commands, markers, clock observations, status,
and buffered watch chunks to the phone. The phone assembles both device streams
and is the only MVP client that uploads research sessions to the backend.
Temporary loss of connectivity must never cause silent sample loss or create a
false claim of synchronization.

### Time and synchronization contract

Every sensor event must retain:

- its device-local monotonic timestamp;
- a per-stream monotonically increasing sequence number;
- sensor type, axes/values, units, and reported accuracy;
- device and sensor identifiers; and
- flags for gaps, discontinuities, or unavailable accuracy.

Each session must retain wall-clock anchors for audit and monotonic clock
observations exchanged between phone and watch. Synchronization processing
estimates offset, delay, and drift without replacing original timestamps.
The research bundle stores raw timestamps, synchronization observations, the
selected correction method and version, and quality statistics. Sessions that
do not satisfy the study's synchronization threshold remain available for
audit but must be flagged or excluded from synchronized analyses.

The acceptable synchronization-error threshold is a feasibility decision for
the successor project. It must be measured on target phone/watch combinations
before a study begins rather than asserted from API documentation.

### Logical research entities

| Entity | Purpose |
|---|---|
| Study | Governance, enrollment window, retention policy, and approved protocol versions |
| Protocol | Ordered instructions, movement vocabulary, device requirements, and quality gates |
| Participant pseudonym | Study-specific identifier separated from direct identity and invitation data |
| Device | Phone/watch model, OS/app version, sensor inventory, and stable study-local identifier |
| Session | Participant attempt, protocol version, arm, device pair, timing anchors, and status |
| Sensor stream | Device, sensor type, units, requested rate, observed timing, and sample reference |
| Event | Immutable timestamped sensor observation with sequence and accuracy metadata |
| Repetition marker | Prompted or participant/researcher-marked movement boundary |
| Annotation | Versioned label, quality judgement, author role, rubric, and confidence |
| Consent record | Version accepted, time, permitted uses, withdrawal state, and audit history |
| Upload | Bundle checksum, transfer state, validation outcome, and retry history |
| Dataset version | Immutable selection of sessions, exclusions, labels, transformations, and provenance |

Identifiers and schemas must be versioned. The successor repository should
publish a machine-readable schema and migration policy before collecting study
data.

### Data lifecycle and backend

The proposed backend is a dedicated REST API described with OpenAPI, backed by
a relational PostgreSQL metadata store and S3-compatible object storage for
immutable recording bundles. An asynchronous validation worker verifies bundle
schemas, checksums, stream ordering, timestamps, marker references, consent
eligibility, and study/device compatibility before admitting a session into a
dataset candidate pool.

The lifecycle is:

```text
phone and watch capture
-> encrypted local buffers
-> watch-to-phone transfer
-> phone-side assembly and validation
-> resumable authenticated upload
-> immutable raw bundle
-> server-side validation and quarantine or acceptance
-> versioned annotation and derived datasets
-> reproducible training and evaluation
```

Raw recordings are append-only. Clock correction, resampling, coordinate
transforms, segmentation, pose landmarks, labels, features, and exclusions are
versioned derived records and must never overwrite raw events. Every dataset
version must identify its source sessions, transformation code/configuration,
annotation versions, exclusions, and checksums.

Immutability means that retained raw data cannot be edited in place; it does not
override a participant's withdrawal or deletion rights. An authorized deletion
removes the complete affected raw bundle and derived references according to
the study policy. Where legally appropriate, the system retains only a
non-identifying audit tombstone showing that a governed deletion occurred.

Authentication must distinguish participant, researcher, annotator, and
administrator roles. Invitation and direct identity data must be separated
from pseudonymous motion data. API authorization, audit records, encryption in
transit and at rest, key management, backups, and deletion workflows are
requirements, not deployment afterthoughts.

### MVP acceptance gates

The collector is ready for a pilot only when it can demonstrate:

- monotonic, non-null native timestamps and sequence numbers for every retained
  stream;
- explicit detection of missing chunks, duplicates, discontinuities, and
  unsupported sensors;
- measured phone/watch clock offset and drift behavior on every supported
  device pair;
- recovery from phone/watch disconnection and interrupted upload without
  duplicating or silently losing events;
- checksum-verified equality between accepted backend bundles and phone-side
  session bundles;
- enforcement of invitation, consent, role, retention, withdrawal, and deletion
  rules;
- immutable raw data and reproducible derived dataset versions; and
- a small supervised pilot reviewed for usability, protocol adherence, data
  quality, privacy, and battery/thermal behavior.

Classification accuracy and coaching quality are not collector-MVP acceptance
criteria.

## Staged successor roadmap

### Stage 1: Paired timestamped collector

Implement and validate the invited-participant phone/watch system described
above. Establish supported hardware, synchronization bounds, failure handling,
privacy controls, and a reproducible export path.

### Stage 2: Dataset expansion and expert annotation

Run controlled Kenpo studies across participants, arms, device pairs, sessions,
and experience levels. Develop an instructor-reviewed rubric for movement
identity, repetition boundaries, execution dimensions, ambiguity, and annotator
confidence. Include natural transitions and non-movement intervals rather than
collecting only ideal pre-segmented repetitions.

### Stage 3: Recognition and repetition segmentation

Train participant-independent models for requested-movement recognition,
continuous repetition segmentation, and uncertainty estimation. Compare new
watch models with Phase 0 findings without presenting cross-device differences
as direct replication.

### Stage 4: Confidence-aware Kenpo feedback

Co-design a limited feedback vocabulary with qualified Kenpo instructors and
learners. Separate motion inference from a rule-constrained coaching policy.
When evidence is missing, ambiguous, or contradictory, request another attempt
instead of producing authoritative advice. Evaluate learning outcomes and human
factors, not classification performance alone.

### Stage 5: Computer vision and multimodal fusion

Add phone-camera pose estimation only after a new consent and privacy review.
Begin with late fusion: independent IMU and vision models expose predictions,
quality indicators, and uncertainty to a fusion layer. This supports IMU-only,
vision-only, multimodal, and abstention behavior and makes disagreement easier
to inspect before considering learned joint representations.

### Stage 6: Curriculum-driven multi-sport platform

Represent a sport as versioned curriculum content: skills, exercise protocols,
sensor requirements, movement phases, evaluation dimensions, model bundles,
and expert-approved feedback rules. Each new sport requires its own experts,
dataset, validation, safety analysis, and learning evaluation. Kenpo evidence
must not be treated as proof of sport-independent validity.

## Scope boundaries

The following are outside the collector MVP:

- computer-vision or video capture;
- real-time classification or coaching;
- technique scores or expertise claims;
- public or unsupervised crowdsourcing;
- automatic clinical, diagnostic, rehabilitation, or injury-prevention advice;
- a universal motion model; and
- support for sports beyond the initial controlled Kenpo protocol.

These capabilities enter the successor roadmap only after the preceding data,
validation, ethics, and human-oversight gates have passed.

## Ethics, safety, and research governance

Every study requires informed consent written for its actual collection and
reuse, a documented retention period, a withdrawal/deletion process, data
minimization, pseudonymization, encryption, role-based access, and an auditable
record of exports and annotations. Adding video is a material protocol change,
not merely another sensor field.

The study's privacy and infrastructure review must also account for the
transport behavior of the selected phone/watch communication service. Wear OS
Data Layer traffic can use a cloud intermediary when a direct connection is
unavailable, even though the channel is encrypted; this must be reflected in
participant information, processor assessment, and deployment controls.

Qualified instructors must define and review movement-quality concepts and
feedback. Models may identify associations or deviations from labelled
examples; they must not claim biomechanical causation, diagnose injury, or give
prescriptive injury advice. Participants must be told when feedback is
automated, what evidence it uses, and when the system is uncertain.

Research evaluation should include data quality, participant-independent
generalization, device and arm effects, subgroup and accessibility concerns,
explanation stability, calibration and abstention, expert agreement, usability,
and measured learning outcomes. A deployable learning system requires evidence
beyond an offline classifier score.

## Handoff decisions for the successor repository

The following choices are fixed by this proposal:

- the successor is a separate project rather than an expansion of the Phase 0
  academic PRD;
- the MVP is an invited-participant, paired phone/watch research collector;
- the phone application uses Flutter and acts as session coordinator and upload
  gateway;
- the watch application uses native Kotlin/Compose for Wear OS;
- native monotonic event timestamps and clock observations are retained;
- the backend uses a custom OpenAPI-described service, PostgreSQL metadata, and
  S3-compatible immutable object storage; and
- coaching, computer vision, public collection, and multi-sport support are
  gated later stages.

The successor project must resolve measured synchronization thresholds,
supported hardware, deployment provider/region, retention periods, consent
wording, annotation rubric, and research approvals before its first participant
pilot.
