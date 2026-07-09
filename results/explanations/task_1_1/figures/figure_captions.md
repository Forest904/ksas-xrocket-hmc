# Task 1.1 Figure Captions

- `sensor_family_contribution`: Mean normalized random-forest native importance by
  sensor family with fold standard deviation.
- `axis_channel_contribution_heatmap`: Mean normalized native importance by sensor
  family and Android device-frame axis.
- `ranked_channel_importance`: Ranked sensor-axis channels with fold standard
  deviation.
- `fold_stability`: Fold-to-fold channel-rank stability using Spearman correlation.
- `ablation_impact`: Macro-F1 drop after removing saved XROCKET feature groups for
  each sensor family.
- `class_specific_channel_profiles`: One-vs-rest random-forest channel profiles by
  movement class.
- `method_agreement`: Agreement between native importance and validation drops for
  channels.
