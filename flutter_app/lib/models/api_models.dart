// API katmanında kullanılan veri modelleri.

class PredictResult {
  PredictResult({
    required this.risk,
    required this.classId,
    required this.threshold,
    required this.explanations,
    required this.topContributors,
  });

  final double risk;
  final int classId;
  final double threshold;
  final List<String> explanations;
  final List<Map<String, dynamic>> topContributors;

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    return PredictResult(
      risk: (json['risk'] as num?)?.toDouble() ?? 0.0,
      classId: (json['class'] as num?)?.toInt() ?? 0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      explanations: (json['explanation'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      topContributors: (json['top_contributors'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
    );
  }
}

class MetricsResult {
  MetricsResult({
    required this.accuracy,
    required this.f1,
    required this.rocAuc,
    required this.threshold,
    required this.modelName,
    required this.featureImportance,
  });

  final double accuracy;
  final double f1;
  final double rocAuc;
  final double threshold;
  final String modelName;
  final List<Map<String, dynamic>> featureImportance;

  factory MetricsResult.fromJson(Map<String, dynamic> json) {
    return MetricsResult(
      accuracy: (json['accuracy'] as num?)?.toDouble() ?? 0.0,
      f1: (json['f1'] as num?)?.toDouble() ?? 0.0,
      rocAuc: (json['roc_auc'] as num?)?.toDouble() ?? 0.0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      modelName: json['model_name']?.toString() ?? '-',
      featureImportance: (json['feature_importance'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
    );
  }
}

