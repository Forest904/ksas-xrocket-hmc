# Human Motion Computing — Detailed Lecture Summary and Part I Assignment

## Source material and course structure

This document summarizes the 72-slide **Human Motion Computing** course taught by Full Professor Olga C. Santos for *Advanced Topics in Computer Science* at Università Roma Tre on June 1 and June 3, 2026. The first lecture introduces Human Motion Computing (HMC) and develops the inertial-signal approach; the second lecture develops computer-vision-based motion modeling and then surveys educational, sports, healthcare, and active-ageing applications. The final section contains only the selected **Part I – Human Motion Computing with Inertial Signals** assignment. All unselected assignment alternatives have been removed so that the project scope is unambiguous.

The course is situated in the work of the Physical User Modelling AI Research Center (PhyUM), which grew from a research line established in 2014, to a laboratory in 2018, and to a research center in 2023. PhyUM combines research, innovation, knowledge transfer, outreach, and training to build personalized intelligent systems for the digital transformation of physical activities. Its guiding perspective is **Hybrid Intelligence**: AI is not treated as a replacement for learners, instructors, clinicians, or other experts, but as a collaborator that augments human observation, interpretation, and decision-making.

---

# Lecture 1 — Foundations of Human Motion Computing

## 1. Human motion as a computational and human phenomenon

Human motion is one of the most fundamental forms of human behavior. People move to learn skills such as sports, music, surgery, and martial arts; to communicate through gesture, sign language, and social interaction; to perform professional and manual work; and to maintain health through physical activity, rehabilitation, and active ageing. Motion is therefore not reducible to generic “physical activity.” A meaningful execution contains coordination, balance, timing, precision, spatial awareness, force regulation, rhythm, and motor control. These capacities are acquired and refined through repeated practice, feedback, and gradual adaptation.

The central premise of HMC is that movement is both an **action** and an **information source**. A motion trace can reveal what a person is doing, how well the action is being performed, the performer’s expertise level, fatigue, stability, emotional state, and aspects of physical or cognitive condition. HMC studies how this information can be sensed, represented, modeled, interpreted, and used to support a person through AI-enabled interventions.

The lectures frame HMC as an extension of intelligent learning environments. Earlier adaptive-learning systems focused mainly on cognitive activity, while affective-computing systems added emotional and physiological information. HMC adds the psychomotor domain: the system must understand the integration of mental processes with muscular activity. This makes motion a first-class component of user modeling, rather than an incidental interaction signal.

## 2. Cognitive, affective, and psychomotor learning

The course uses Bloom-related taxonomies to distinguish three learning domains. The **cognitive** domain concerns intellectual behavior and thinking. The **affective** domain concerns feelings, attitudes, opinions, and values. The **psychomotor** domain concerns the acquisition of skills that require coordinated mental and muscular activity. Several psychomotor taxonomies—Dave, Harrow, Simpson, Thomas, and Ferris and Aziz—describe learning as progression from observation and guided imitation toward confident, coordinated, adaptable, and ultimately original performance.

Simpson’s taxonomy is particularly useful for interpreting intelligent psychomotor systems. It moves through perception of sensory cues, readiness to act, guided response, mechanism, complex overt response, adaptation, and origination. The important transition is from **guided response**, where the learner imitates and relies on trial and error under supervision, to **mechanism**, where the learned response becomes habitual and can be executed confidently without constant external guidance. Higher stages add highly coordinated performance without hesitation, adaptation of existing patterns to special requirements, and the creation of new movement patterns for novel situations. This taxonomy clarifies that a system which merely makes a user copy an avatar may support early-stage guided response but does not automatically demonstrate learning, internalization, transfer, or mastery.

## 3. Goals of intelligent systems that consider motion

The course identifies four broad goals for motion-aware intelligent systems. First, **psychomotor learning and training** aims to improve the performance of motor skills. Second, **cognitive or embodied learning** uses movement to support conceptual understanding; physical action can become a mediator for learning abstract ideas. Third, **emotional management** examines how emotions affect posture, body motion, and displacement, and how feedback might help users regulate those states. Fourth, **health-oriented support** includes postural correction, injury prevention, rehabilitation, and active ageing.

To achieve these goals, a learner’s physical actions must be monitored, compared with an expected execution, and corrected when needed. This creates a closed-loop architecture rather than a one-off recognition task. The system observes behavior in real time, constructs a computational model, diagnoses differences between current and desired performance, decides what intervention is appropriate for this person and context, and delivers feedback through a suitable modality.

## 4. The SMDD framework

The core architecture is the **SMDD framework**:

1. **Sensing the movement** captures the learner’s motion and context using inertial sensors, cameras, physiological devices, or combinations of them.
2. **Modeling the movement** transforms raw sensor streams into computational representations, such as multivariate time series, body keypoints, joint angles, movement phases, or semantic descriptions.
3. **Designing the feedback** diagnoses the gap between the learner’s execution and an expert or clinical reference and selects a pedagogically and ethically appropriate intervention.
4. **Delivering the feedback** presents support in real time through visual, auditory, or tactile channels.

The four components are cyclic. Feedback changes the next execution, which is sensed again, so the system can update its model and progressively personalize support. A complete HMC system therefore requires more than a classifier: it needs a sensing strategy, motion representation, user model, context model, error diagnosis, feedback policy, delivery mechanism, and evaluation of the resulting learning or health experience.

## 5. Why martial arts are a productive HMC domain

Martial arts are used throughout the course as a research testbed because they are long-established, codified systems of techniques and tactics. They contain predefined movements structured by levels, are governed by physical laws, and provide recognizable reference patterns against which performance can be compared. Their techniques range from relatively static postures, such as some Tai Chi forms, through quick strikes and blocks in Karate or Kenpo, to complex defensive responses and circular interactions in Aikido.

Martial arts also have broader benefits: cardiovascular fitness, muscular elasticity, concentration, stress reduction, discipline, security, and potentially reduced violence. The movements share characteristics with many other physical activities, so the computational methods can transfer to rehabilitation, exercise, sports coaching, music performance, and occupational skills.

Earlier technological systems often used optical capture to make learners imitate an instructor’s posture or gesture and then gave generic visual feedback. The limitation was not only recognition accuracy but **lack of personalization**. Different learners have different body dimensions, abilities, experience, affective responses, goals, preferences, and safe ranges of motion. A useful HMC system must distinguish an error that reflects lack of skill from a harmless individual variation, and it must choose feedback that helps rather than distracts, overloads, frustrates, or endangers the learner.

---

# Lecture 1 — AI Support, Human-Centered Design, and Motion Representation

## 6. Complementary sensing and modeling modalities

The lectures distinguish three major sensing perspectives. **Inertial measurement units (IMUs)** capture acceleration and rotation. They are portable, can operate without a camera, and preserve more privacy, but provide no direct visual context. **Cameras** capture posture and spatial motion, yielding rich full-body information, but depend on lighting, line of sight, background, occlusion, and camera placement. **Physiological sensors** capture internal states such as electrodermal activity or heart-related responses. These can explain effort, stress, or arousal but provide limited direct information about motion geometry.

Because each modality sees a different aspect of behavior, the course emphasizes **multimodal data fusion**. Inertial and visual data can be fused before modeling, or separate models can be built and their outputs combined. Physiological data can add context to either approach. The choice depends on the application: a camera may be best for full-body geometry, an IMU for hidden or prolonged movement, and a physiological sensor for internal state. A sophisticated system may use all three, but additional modalities also increase setup burden, synchronization complexity, privacy risk, and the difficulty of explaining model decisions.

## 7. Signal-processing and computer-vision pipelines

The inertial pipeline begins with continuous multichannel time-series data. It usually requires segmentation, feature preparation, normalization, and sometimes dimensionality reduction. Features can be extracted in the time domain, frequency domain, or time-frequency domain through wavelets. Machine-learning or deep-learning algorithms then identify movement units, classify activity or expertise, and evaluate performance against expert-labeled examples.

The computer-vision pipeline begins with video frames. A human-pose-estimation algorithm extracts body landmarks, creating a skeletal representation in 2D image coordinates and sometimes approximate 3D coordinates. The data then undergo filtering, preprocessing, and normalization. Features may describe static limb positions, joint angles, or dynamic heatmaps and trajectories. Rule-based reasoning, machine learning, deep learning, or state machines can then identify actions and evaluate movement quality.

The main architectural lesson is that raw data are not meaningful by themselves. A signal sample, pixel, or keypoint becomes useful only when placed within a representation that connects to the motion, the user, and the purpose of the system.

## 8. Hybrid Intelligence and the human in the loop

The **Hybrid Intelligence** paradigm requires HMC systems to be collaborative, adaptive, responsible, and explainable. Collaboration means creating useful synergies among learners, instructors, clinicians, data scientists, psychologists, and developers. Adaptation means learning from different bodies and environments rather than forcing everyone into a single template. Responsibility covers privacy, safety, fairness, and ethical handling of sensitive biometric data. Explainability enables dialogue: users and experts need to understand what data were used, why the system reached a conclusion, and how a recommendation relates to the intended learning or health objective.

The CARAIX perspective extends the SMDD process with human-centered engineering and explicit stakeholder involvement before, during, and after modeling. Human experts participate in data description and model design; explainability methods such as LIME, SHAP, or class-activation approaches can help diagnose decisions; and post-deployment explanation, dissemination, perturbation testing, and reproducibility are treated as part of the system rather than optional extras. The course repeatedly warns that an accurate but opaque model may be unusable when it judges a person’s body, skill, or health.

## 9. Semantic descriptions of movement

A recurring challenge is the gap between low-level kinematics and human-understandable movement concepts. The **MoRTE-Laban** work illustrates a neurosymbolic approach based on Labanotation and Laban Movement Analysis. Individual motion segments are detected from skeletal positions, annotated with biomechanical and symbolic information, exported in XML, and associated with recognizable actions. The framework supports bidirectional conversion between kinematic and symbolic representations: a recorded pose sequence can be translated into a semantic description, and a symbolic description can be mapped back into a 3D movement sequence.

This bidirectionality improves interpretability and reproducibility. A user or expert can reason about direction, level, body part, timing, and movement quality instead of inspecting raw coordinates. The same representation can support explanation, comparison, and—potentially with generative AI—the creation of new movement examples. The larger lesson is that semantic abstraction is not decorative metadata; it is a bridge between numerical modeling and domain reasoning.

## 10. Multisensorial and personalized feedback

Feedback can be visual, auditory, tactile, or multimodal. Vibrotactile support is especially valuable when the user should not look away from the activity or when feedback must be associated with a specific body part. A complete intervention policy asks what feedback to give, how much, where on the body, when to deliver it, to whom, through which channel, and under which contextual conditions. The system must evaluate not only motion accuracy but also the personalized learning experience: whether feedback is timely, understandable, non-intrusive, safe, motivating, and effective.

Generative AI may help formulate feedback content, but the course places this within an expert-governed pipeline. Generated feedback must be constrained by validated movement models, user characteristics, pedagogical goals, and ethical rules. Fluent language is not evidence that a recommendation is biomechanically correct or clinically safe.

---

# Lecture 1 — Human Motion Computing with Inertial Signals

## 11. What inertial signals represent

Inertial sensing represents motion as multivariate temporal data. Accelerometers measure linear acceleration, gyroscopes measure angular velocity, and magnetometers provide orientation information relative to the Earth’s magnetic field. A smartphone or wearable can therefore produce three, six, or nine channels depending on the available sensors. The signals are portable and can be collected over long periods or in locations where cameras would be inconvenient, intrusive, or unable to see the relevant body part.

The standard processing flow is: acquire continuous signals, segment them into meaningful movement units, normalize and homogenize sequence length, prepare features or learned representations, train a model with expert labels, and use the resulting system for activity recognition, expertise classification, or performance assessment. Segmentation is crucial because the same recording may contain pauses, transitions, different phases, and “no movement” intervals.

## 12. KSAS: smartphone sensing for Kenpo blocking movements

The **KSAS** work uses a mobile application to support practice of American Kenpo Karate Blocking Set I. The set contains an initial position and five principal arm blocks: upward block, inward block, extended outward block, outward downward block, and rear elbow block. The application was designed to recognize the movements and provide auditory and haptic feedback, with positive feedback for a correct movement and corrective signals for an incorrect one.

The dataset described in the slides was collected from 20 participants and includes executions with both arms. Preprocessing segmented the time sequence into the five movements plus no-movement examples, smoothed the signals using exponentially weighted moving averages, normalized values to the interval from 0 to 1, padded sequences to a common maximum length, and assigned labels. The resulting representation used 18 features—six sensor types/streams across three axes—with examples padded to 56 samples. In the stated configuration, 240 examples were represented as a tensor with dimensions corresponding to examples, channels/features, and sequence length, and the classifier distinguished six classes.

The KSAS case demonstrates the full engineering chain from a wearable sensor to a learning application. It also reveals practical difficulties: pauses may be easier to segment when the order is known; expert and beginner executions can differ in duration; sensor orientation and handedness matter; smoothing can remove noise but also suppress discriminative micro-movements; and small participant numbers can produce optimistic results if data from the same person appear in both training and test sets.

## 13. Phy+Aik: motion data for expertise and embodied learning

The **Phy+Aik** projects used smartphones or inertial sensors to study Aikido movements such as *shikko* knee walking and attacks with a wooden sword. The acceleration signal visibly differentiated novice from expert motion in both time and frequency domains. Experts showed more regular and controlled patterns, while novices produced noisier and less stable traces. The concept of the *hara*—the body’s center of mass and stability—was used to connect sensor patterns to martial-arts biomechanics.

The same motion context was used for embodied learning of physics. Aikido movements illustrated force, tangential velocity, torque, and angular momentum. The educational premise is that movement, thinking, and learning are interconnected: physically enacting a rotational or force-related concept can reinforce an abstract equation. This is an example of learning **with** motion, rather than merely learning **about** how to perform the movement.

## 14. iELA and the modeling of expertise

The **iELA** line asks whether inertial data can reliably model user expertise while psychomotor skills are being acquired. Martial arts provide explicit competence labels through belt levels, allowing a binary distinction between beginners and experts for modeling purposes. The goal is not only to classify expertise but to enable personalized interventions based on the movement characteristics associated with different proficiency levels.

Three datasets are highlighted:

- **D1, Bokken/Shomenuchi:** 153 participants, three accelerometer axes, and a blade-swing movement. Segmentation was limited.
- **D2, Shikko:** 185 participants, six axes from accelerometer and gyroscope, with seven phases—two outward/going phases, three turning phases, and two return phases—and two broad movement types, straight and turn.
- **D3, Kenpo Blocking Set I:** 16 participants, nine axes from accelerometer, gyroscope, and magnetometer, with six right-hand movement units.

These datasets vary in participant count, sensor configuration, rotational content, movement complexity, and availability of magnetometer data. That variation is important because no representation or classifier is universally optimal.

## 15. Coordinate transformations and sensor fusion

A major research question is whether transformations of inertial data improve proficiency assessment. Raw Cartesian coordinates `(x, y, z)` may not align with the geometry of a movement. Spherical coordinates can better represent radial and angular structure; cylindrical coordinates can better represent rotations around an axis; and quaternions provide a compact representation of 3D orientation and avoid some singularities associated with Euler angles.

The course links representation choice to movement nature. Martial-arts actions can be linear, circular, or helicoidal. A cylindrical or spherical transform may expose discriminative structure that is dispersed across Cartesian axes. Quaternion estimation requires six or nine sensor axes and can be computed with analytical methods or filters such as Madgwick, Mahony, and Kalman. Magnetometer availability influences filter quality: the analytical approach may omit it, Madgwick and Mahony can treat it as optional, and the Kalman configuration discussed requires it.

The experiments generated 33 dataset variants by combining original coordinates, geometric transformations, sensor-fusion methods, whole movements, and segmented phases. This is an important methodological point: representation engineering is itself part of the model. A “raw signal” baseline should be compared with transformations that have a biomechanical rationale, rather than assuming a more complex representation must always improve accuracy.

## 16. TSFEL, ROCKET, classifiers, and sequence handling

Two broad feature approaches were compared. **TSFEL** extracts explicit temporal, statistical, and spectral variables from time series; LASSO can reduce dimensionality and retain a compact subset. This approach is interpretable, relatively fast, and suitable for hardware-limited deployment. **ROCKET** applies many random one-dimensional convolutional kernels to discover patterns in time series. It is designed to be fast despite using a large feature map and is often paired with linear classifiers.

The models were evaluated with Random Forest, Logistic Regression, Ridge Classifier with cross-validation, stochastic gradient descent, and k-nearest neighbors. Sequence lengths were homogenized through resampling. Whole-movement models were compared with segmented models, including the seven phases of *shikko* and the partial movements of the Kenpo set. Min-max normalization was used before classification.

The course stresses that modeling choices interact. Random Forest is robust and easy to use with explicit features; linear models are recommended with ROCKET-derived features; SGD benefits from many samples; and kNN can be robust to noise with a sufficiently large training set but is costly at inference. Evaluation should consider participant-independent splits, class balance, temporal leakage, and the difference between recognizing a movement and judging its quality.

## 17. Reported iELA outcomes

For D1, ROCKET outperformed TSFEL even in its weaker configuration, with best results around 67% for Cartesian or cylindrical representations, compared with about 61% for a TSFEL/kNN combination. D2 produced stronger results. TSFEL with Random Forest achieved values in the mid-to-high 70% range, while ROCKET on complete spherical-coordinate sequences reached approximately 82% with Ridge or SGD. In D3, TSFEL and ROCKET reached about 80% in selected complete-movement configurations, and quaternion-based representations were competitive or beneficial when magnetometer information was available.

The derived conclusions are nuanced. For rotational movements, spherical, cylindrical, or quaternion representations can improve on raw Cartesian data. Quaternions were especially helpful in D3, where magnetometer data supported orientation estimation. TSFEL remained attractive for embedded hardware because it was faster and often close to ROCKET. Random Forest performed well across many explicit-feature configurations. However, performance alone did not answer the central interpretability question: **which channels, temporal scales, and biomechanical characteristics actually drove the decision?**

## 18. From ROCKET to X-ROCKET

ROCKET typically uses thousands of randomly generated convolutional kernels. Its speed and accuracy are useful, but random kernels make feature importance unstable across executions. **X-ROCKET** replaces this with fixed kernels to improve explainability. It can report which sensor channels are influential and which dilation values—spacing between elements of a convolutional filter—capture discriminative temporal structure.

Dilation acts like a “time zoom.” Low dilation means the kernel examines nearby samples, making it sensitive to short-duration, high-frequency events such as sudden changes, technical micro-adjustments, rhythm, and precision. High dilation examines samples farther apart, capturing long-duration, low-frequency structure such as posture, global flow, stamina, and consistency throughout an exercise. Human expertise is therefore multiscale: an expert may be precise at the millisecond scale while also maintaining stable form over a minute.

## 19. Explainable expertise: force and anthropometry in Shikko

The force-aware Aikido study combined motion features with body mass. Lateral acceleration on the Y axis was influential for some kernels, suggesting that experts reduce unnecessary horizontal bounce during knee walking. Force, computed from mass and acceleration magnitude, emerged as a stable indicator; in the reported interpretation, acceleration magnitude itself was less relevant, so the body-mass component carried substantial discriminative information.

This result illustrates both the value and danger of explainability. A biomechanical interpretation—experts stabilize lateral movement—is plausible and useful for feedback. At the same time, a model may classify body size rather than skill if anthropometric information correlates with labels. Explanations must therefore be examined for confounding, kernel stability, and fairness. A feature being important does not automatically mean it is pedagogically legitimate.

## 20. Explainable sleep-motion analysis

A second X-ROCKET application used wrist-worn inertial sensors as a non-invasive, cost-effective tool for differentiating early cognitive decline from advanced dementia through nocturnal movement. The model emphasized high dilations, meaning low-frequency patterns were more discriminative than quick twitches. The interpretation was that the model captured long-term postural shifts and disruptions related to REM atonia rather than isolated high-frequency events.

Channel analysis suggested that Y and Z combinations were relevant, indicating coordinated movement in two directions, while rotations associated with wrist supination and pronation were less important. This case shows why temporal scale depends on the domain: in skilled martial-arts movement, both micro-adjustments and global consistency matter; in overnight health monitoring, long-duration postural organization can dominate.

## 21. Inertial-signal conclusions

Inertial processing turns raw accelerometer, gyroscope, and magnetometer signals into evidence about activity, technique, expertise, or health. Wearables can sense “invisible” properties that cameras may miss, such as internal stability, micro-rhythms, hidden-joint motion, or patterns that unfold during sleep. The most useful coordinate system is not always Cartesian; representations should align with the movement’s geometry. Finally, mastery and health are multiscale phenomena, so interpretable temporal models must account for both short and long patterns.

---

# Lecture 2 — Human Motion Computing with Computer Vision

## 22. Human pose estimation as a structured motion representation

Human Pose Estimation (HPE) transforms an image or video into a structured representation of the body. Instead of modeling raw pixels directly, the system locates anatomical keypoints and connects them into a skeleton. Two-dimensional keypoints use image coordinates; three-dimensional systems add measured or estimated depth. The COCO format with 17 keypoints is a common benchmark representation, although individual algorithms may use more landmarks or include hands and face.

A typical pipeline records video, extracts landmarks, filters and normalizes the keypoints, computes static or dynamic features, and applies an AI model. Static features include joint angles and limb positions. Dynamic features include trajectories, velocity, phase transitions, and movement heatmaps. The output can recognize an action, classify quality, compare a learner with an expert, or populate clinical scales.

The camera-based modality is non-intrusive and spatially interpretable. It is particularly valuable for full-body biomechanics, posture, and interactions among multiple people. Its limitations are equally important: body parts can be occluded, lighting and background affect detection, depth is often approximate, and a fixed line of sight may be difficult in a home.

## 23. MediaPipe Pose and KLS

**MediaPipe Pose** follows a detect-then-track strategy. A person is detected, a region of interest is established, 33 BlazePose landmarks are estimated, and subsequent frames track those landmarks efficiently. Its computational cost is low, it runs well on mobile or CPU hardware, and it offers approximate 3D information. It is therefore well suited to interactive, real-time, single-person applications.

The **Kenpo Learning Simulator (KLS)** extends the KSAS blocking-set work into a virtual-reality and AI learning environment. A learner sees an avatar or reference execution, performs the movement, and receives assessment. The system uses MediaPipe Pose to observe the body and a state-machine representation to define expected movement stages and transitions. This design is interpretable: experts can specify valid states, joint or positional constraints, and error conditions without requiring a large labeled dataset for every possible execution.

The architecture connects to Simpson’s psychomotor taxonomy. Guided practice may support imitation, but the research question is whether repeated guided execution produces recall, internalization, and independent performance. A rule-based finite-state model is valuable when the target movement has known phases and biomechanical constraints, when errors need to be explained in domain terms, and when training data are limited. Its weakness is brittleness: rules can be difficult to scale to unconstrained styles, body diversity, and unanticipated transitions.

## 24. Emotion recognition under partial occlusion

KLS-related work also examined classification of facial emotion when part of the face is occluded by XR headsets or masks. Partial occlusion removes information but does not make classification impossible. The mouth was especially informative for neutral expression, happiness, and anger, which made headset occlusion less damaging than face-mask occlusion for those emotions. The eyes were more informative for surprise.

This study illustrates a general HMC lesson: the sensing apparatus changes what can be inferred about the user. A VR headset may improve immersion while hiding affective cues; a mask may preserve the eyes while obscuring the mouth. Multimodal or context-aware systems should account for these systematic losses rather than treating missing facial regions as random noise.

## 25. OpenPose and KUMITRON

**OpenPose** uses a bottom-up strategy. It first detects body parts across the image and then uses Part Affinity Fields (PAFs) to associate those parts into individual skeletons. It supports body, face, and hand keypoints and performs particularly well in multi-person scenes, though it has high computational cost and generally prefers GPU hardware.

The **KUMITRON** system applies pose estimation to Karate kumite, where two practitioners interact, overlap, move rapidly, and continually change relative position. A bottom-up approach is well matched to this setting because it does not begin by isolating each complete person and then estimating pose. Instead, it detects evidence for all body parts and assembles the people, which can be more robust when bodies are close, partially occluded, or interacting. The resulting features feed machine-learning or deep-learning models for technique and performance evaluation.

KUMITRON also explores collaborative learning. In combat or paired practice, performance is relational: distance, timing, attack-defense coordination, and interaction patterns matter. A model should therefore avoid evaluating each skeleton as if the other person were merely background. Multi-person pose estimation enables features that describe both individual technique and the dynamics between practitioners.

## 26. YOLO Pose and SPAF

**YOLO Pose** uses a single-stage strategy that detects people and keypoints simultaneously. It commonly outputs the 17 COCO keypoints, has high speed, medium computational cost, and can run on CPU or GPU. It is attractive for deployment because the detection and pose-estimation tasks are integrated.

The **SPAF** work applies YOLO Pose to personalized physiotherapy. Its central contribution is a **semantic layer** between raw keypoints and clinical assessment. Layer 1 detects keypoints. Layer 2 converts them into motion metrics such as ranges of motion, joint speed, gait patterns, postural deviations, and task-specific parameters. Layer 3 maps those metrics into established clinical scales and tables, such as TCT, NIH-related measures, Berg, Barthel, and other functional-assessment instruments.

Pose estimation alone is insufficient for clinical interpretation because coordinates do not answer clinical questions. A clinician needs to know whether the patient can maintain trunk control, whether a limb moves through a safe and meaningful range, how quickly or symmetrically a task is performed, and how those observations contribute to a validated scale. The semantic layer makes the system algorithm-independent at the sensing level and domain-specific at the interpretation level. It also makes results auditable: a score can be traced back to measured movement parameters rather than appearing as an opaque label.

## 27. Comparison of the three main HPE approaches

The slides summarize the trade-offs as follows:

| Feature | MediaPipe Pose | OpenPose | YOLO Pose |
|---|---|---|---|
| Keypoints | 33 BlazePose landmarks | 25 body points plus face and hands | 17 COCO keypoints |
| Core strategy | Detect once, then track | Bottom-up assembly using PAFs | Single-stage person and keypoint detection |
| Multi-person capability | Limited | Excellent | Good |
| Speed / real-time suitability | Very high | Medium to low | High |
| 3D support | Approximate 3D | Mainly 2D | Mainly 2D |
| Preferred hardware | Mobile / CPU | GPU | CPU / GPU |
| Computational cost | Low | High | Medium |
| Ease of use | Very high | Low | High |
| Best fit in course examples | Interactive applications | Interaction analysis | Deployment and scalable assessment |

No algorithm is universally best. MediaPipe is a strong choice for low-friction home or educational interaction with one user. OpenPose is justified when rich keypoints and multi-person association are central. YOLO Pose is attractive for fast integrated detection and deployment, especially when a semantic layer will transform its keypoints into application-specific meaning. OpenCV-based approaches can provide additional baselines or services, but their suitability must be evaluated under the same conditions rather than assumed from a library label.

## 28. Computer-vision conclusions

HPE converts images into structured body models that can be processed through symbolic rules, state machines, machine learning, or deep learning. The three PhyUM systems—KLS, KUMITRON, and SPAF—were developed in education, sport, and healthcare, but they solve the same core problem: automatically evaluate the quality and meaning of human movement from visual observations. Their differences are architectural responses to different constraints: real-time single-person interaction, close multi-person combat, and clinically meaningful assessment.

---

# Lecture 2 — Applications of Human Motion Computing

## 29. The 3D-IWA Motion-for-Learning model

The education application is organized by the **3D-IWA Motion-for-Learning model**, which distinguishes learning **IN**, **WITH**, and **ABOUT** motion.

**Learning IN Motion** treats movement as the context or condition of learning. The learner may walk while brainstorming, take an active break, or move around a simulated environment. Motion affects cognition, creativity, engagement, stress, and collaboration, even though the movement itself is not the learning objective.

**Learning WITH Motion** treats movement as a mediator or pedagogical resource. Embodied gestures help the learner understand another topic, such as using Aikido actions to experience force, torque, or angular momentum.

**Learning ABOUT Motion** treats physical performance as the content and objective. The learner is acquiring a motor skill such as a martial-arts technique, sign, musical posture, basketball action, workout exercise, or rehabilitation task.

This model prevents conceptual confusion. A motion-aware educational system should state whether it is using movement to create a learning context, to explain something else, or to teach the movement itself. The sensing, evaluation metrics, and feedback design differ across these categories.

## 30. PhyUM educational and training systems

**EMo2Cla** is an example of learning in motion. Nursing students move around a patient simulator during emergency-room scenarios. Localization sensors, microphones, and physiological wristbands capture teamwork, movement, and stress reactions. The goal is to adapt and improve teaching and learning that occurs physically in the classroom, not simply to classify a single gesture.

**Phy+Aik** exemplifies learning with motion by using martial-arts movement to reinforce physics concepts. The learner’s body becomes a manipulable model of abstract mechanics.

Several systems exemplify learning about motion. **iELA**, **KSAS/KLS**, and **KUMITRON** support martial-arts technique and expertise. **AHTROM** corrects body posture for trombone playing to improve breathing and avoid injury, using a vibrotactile device. **SLS-T** performs automatic segmentation of signs versus no-sign intervals in sign-language speech and can support learning how to execute each sign. **iBAID** analyzes agility tests and free throws to recommend basketball activities that improve technique, prevent or support recovery from injury, and sustain activity with age. **MyWorkout** uses MediaPipe Pose and finite-state models to provide real-time gym feedback and performance analysis.

These systems show that HMC is not restricted to elite sport. The same computational concepts—phase segmentation, posture constraints, trajectories, expertise models, error diagnosis, and feedback—apply to music, sign language, exercise, classroom simulation, and healthcare.

## 31. Extending learning-management systems with sensor data

The course also considers how HMC can be integrated into online education rather than isolated in a standalone app. **LA-ReflecT** is an authoring platform that connects learning-management systems such as Moodle or Sakai with sensor-rich learning tasks, activity-attempt interfaces, reflection dashboards, and a learning record store. A teacher can define an activity, associate it with external sensor data, and make motion analytics part of the learning record.

This integration raises standardization and pedagogy questions. Sensor data must be synchronized with learning events, represented in interpretable metrics, and connected to reflection rather than shown as unexplained charts. The objective is to extend the LMS with evidence of embodied activity while preserving a coherent learning design.

## 32. Active ageing: FRAGILESS and MITAICHI

The active-ageing application includes **FRAGILESS**, a 3D exergame with dynamically adapted difficulty intended to reduce frailty in older adults. It models motor activity and adjusts challenge so that the game remains achievable, safe, and beneficial.

**MITAICHI** guides users through Tai Chi postures using visual and haptic feedback, including inertial sensors in hand controllers to monitor alignment. Tai Chi is framed as beneficial for mobility, balance, autonomy, and well-being. The system illustrates how cameras, controllers, and feedback modalities can be combined, but it also highlights deployment constraints: older users may have limited digital literacy, reduced mobility, different sensory capabilities, and little tolerance for complex calibration.

## 33. The EMERGE methodology and design constraints

The **EMERGE** methodology is a multi-stage participatory approach that uses theatre and action to elicit users’ tacit knowledge. Instead of asking older adults only abstract questions about a hypothetical system, participants enact situations, try prototypes, and reflect with stakeholders such as gerontologists, technical teams, and social workers. The stages include theatre-based exploration, enacted co-design, experiential try-out, and stakeholder reflection.

The resulting design constraints are central to any home-based HMC system for older adults. Setup and calibration should be minimal because digital literacy may be low. Home environments are uncontrolled, with variable space, lighting, camera placement, furniture, and clothing. Privacy should be protected by avoiding intrusive sensing where possible. Interaction should be lightweight, comfortable, non-invasive, and robust to physical diversity. A technically accurate HPE model that requires precise camera placement or repeated troubleshooting may fail in practice.

---

# Course Synthesis and Design Principles

## 34. Selecting the right modality

The final synthesis contrasts inertial and visual sensing. Inertial data are temporal waveforms collected by wearables or embedded platforms. They can operate anywhere without line of sight and are particularly appropriate for continuous health tracking, sleep, hidden joints, and subtle micro-movements. Computer vision produces spatial keypoints from external cameras or webcams. It requires acceptable lighting and line of sight but supports interpretable full-body biomechanics, posture analysis, and spatially complex sport or rehabilitation tasks.

Selection should begin with the question, not the algorithm. A designer should ask what movement property must be observed, over what time scale, in which environment, with how many people, and under what privacy and usability constraints. A home Tai Chi system may favor camera-based pose estimation for full-body posture, but an unobtrusive wrist or hand IMU could complement it when hands leave the camera view or when rotational information is important. Multimodal sensing is valuable only when its added information justifies its added burden.

## 35. The Hybrid Intelligence ecosystem

The course’s final ecosystem includes four actors. The learner performs a physical action that is sensed by hardware. The AI engine models the raw data and compares it with expert or clinical knowledge. A human expert oversees ethical parameters, contributes domain knowledge, and validates the system. The system returns explainable, multimodal feedback to the learner. True Hybrid Intelligence means that machines process data while humans guide the pedagogy and meaning.

This structure resists two common mistakes. The first is treating the expert merely as a label provider and excluding them from model design. The second is treating an algorithmic score as an objective truth. Human expertise remains necessary to define what counts as safe, correct, meaningful, and fair.

## 36. Ethical imperative and CARAIX

HMC processes high-risk biometric information. The course’s ethical framework requires four properties. **Collaborative** systems augment instructor or clinician expertise. **Adaptive** systems personalize to the user’s physical, cognitive, and affective needs. **Explainable** systems justify why a movement was flagged and show the evidence behind feedback. **Responsible** systems protect privacy, physical safety, and ethical handling of inertial and biomechanical data, with attention to relevant regulation.

Ethics is therefore an engineering requirement. Camera data may expose a home environment; body features may reveal health or disability; force or anthropometric variables may create unfair expertise predictions; and incorrect feedback may cause injury. Data minimization, secure storage, consent, participant-independent validation, confidence-aware feedback, human override, and transparent limitations belong in the system architecture.

## 37. Integrated takeaways

The course can be summarized through several connected propositions. Human movement is a rich, multiscale signal that reflects action, skill, condition, and context. Raw sensing becomes useful only through an appropriate representation, and representation should align with movement geometry and domain meaning. Inertial sensing excels at continuous temporal evidence; computer vision excels at spatial body structure; physiological sensing adds internal state. Explainability must connect model features to biomechanics or clinical semantics, not merely display feature scores. Feedback must be personalized, multisensorial when useful, and evaluated for learning or health impact. Finally, successful systems are hybrid: AI supplies scalable measurement and pattern recognition, while human experts and users define objectives, validate meaning, and govern ethical use.

---

---

# Selected Assignment Scope — Part I Only

> **Scope note:** The project is exclusively **Part I – Human Motion Computing with Inertial Signals**. The wording of its background, links, tasks, discussions, and deliverable below is preserved from the assignment document; all option-selection material and unselected assignments have been omitted.

```text
Assignment for "Human Motion Computing"
Due: July 20th, 2026

Selected project scope: Part I – Human Motion Computing with Inertial Signals

Generative AI Policy

• Generative AI Tools CAN be used to support the usage of software tools in Part I.
  You should indicate the tools used and for what purpose.

Support: You can contact professor Olga Santos (ocsantos@dia.uned.es) for
questions, hints or suggestions about the assignment.


Part I – Human Motion Computing with Inertial Signals

Background

The KSAS dataset [1] contains smartphone inertial sensor data collected during the
execution of American Kenpo Karate Blocking Set I. The XROCKET algorithm [2] was
used in [3] to classify Aikido practitioners’ expertise while providing interpretable
explanations of the classification decisions. The same approach was also followed in
[4] to differentiate early stages of cognitive decline and advanced dementia based on
the movements performed while sleeping.

Your task is to apply XROCKET to the KSAS dataset (or any other dataset you know or
create) to extract meaningful information from the motion data collected. Details about
the dataset are available in [5]. The implementations done by PhyUM researchers are
available in [6] and [7] as support (but have not been exhaustively tested).

Links

[1] KSAS Dataset: course-provided local dataset.
[2] Time series classification with XROCKET: https://dida.do/blog/explainable-time-series-classification-with-x-rocket
[3] UMAP 2026 paper (to be published in UMAP proceedings: umap26-45.pdf)
[4] Sleep analysis paper: https://link.springer.com/article/10.1007/s10796-026-10736-0
[5] Aikido paper: https://link.springer.com/article/10.1007/s11257-024-09393-2
[6] Code for Sleep analysis: https://github.com/Physical-User-Modeling-PhyUM/EADS
[7] Code for Aikido analysis: https://github.com/Physical-User-Modeling-PhyUM/UMAP26_SP1 (to be available during UMAP conference)


Task 1.1 – Sensor Axes contribution analysis (Where is the information?)

Analyze the XROCKET explanations regarding the information about the IMU axes and
respond:

• Which sensor signals contribute most to classify the user movement and/or
  user performance?

Discussion: Relate your findings to the biomechanics of the movements.


Task 1.2 – Temporal Pattern Analysis: Dilations/Frequencies (At what temporal
scale is the information?)

Analyze the dilation values selected by XROCKET. Determine whether the
classification relies primarily on:

• short-duration, high-frequency motion patterns,
• long-duration, global movement structures,
• or a combination of both.

Discussion: Explain how temporal scale influences movement recognition.


Task 1.3 – Explainable Human Motion Computing

Interpret the most discriminative patterns identified by XROCKET in the dataset used.

Discussion:

• what aspects of the movement are being captured,
• whether the explanations appear meaningful from a human perspective,
• how these explanations could support learning and performance assessment.


Deliverable

Technical Report (PDF). The report should include:

• Description of the inertial dataset used.
• Description of the steps followed to answer the questions.
• Problems encountered (if any) preparing the implementation.
• Link to the code repository, notebook, etc. produced.
• Responses to each of the tasks, including visualizations computed for the
  analysis to support the discussion of the findings.
• Use of Generative AI tools to support the software implementation.
```
