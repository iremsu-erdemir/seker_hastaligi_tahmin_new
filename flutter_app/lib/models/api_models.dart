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
    required this.balancedAccuracy,
    required this.precisionMacro,
    required this.recallMacro,
    required this.threshold,
    required this.modelName,
    required this.featureImportance,
    required this.models,
    required this.preprocessing,
    required this.classificationReport,
    required this.featureImportanceByModel,
    required this.cvTrainRocAucMean,
    required this.cvTrainRocAucStd,
    required this.generatedAt,
    required this.charts,
  });

  final double accuracy;
  final double f1;
  final double rocAuc;
  final double balancedAccuracy;
  final double precisionMacro;
  final double recallMacro;
  final double threshold;
  final String modelName;
  final List<Map<String, dynamic>> featureImportance;
  final List<Map<String, dynamic>> models;
  final List<String> preprocessing;
  final String classificationReport;
  final Map<String, dynamic> featureImportanceByModel;
  final double cvTrainRocAucMean;
  final double cvTrainRocAucStd;
  final String generatedAt;
  final List<ChartAsset> charts;

  factory MetricsResult.fromJson(Map<String, dynamic> json) {
    return MetricsResult(
      accuracy: (json['accuracy'] as num?)?.toDouble() ?? 0.0,
      f1: (json['f1'] as num?)?.toDouble() ?? 0.0,
      rocAuc: (json['roc_auc'] as num?)?.toDouble() ?? 0.0,
      balancedAccuracy: (json['balanced_accuracy'] as num?)?.toDouble() ?? 0.0,
      precisionMacro: (json['precision_macro'] as num?)?.toDouble() ?? 0.0,
      recallMacro: (json['recall_macro'] as num?)?.toDouble() ?? 0.0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      modelName: json['model_name']?.toString() ?? '-',
      featureImportance: (json['feature_importance'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
      models: (json['models'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
      preprocessing: (json['preprocessing'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      classificationReport:
          json['classification_report_test_best']?.toString() ?? '',
      featureImportanceByModel: Map<String, dynamic>.from(
          json['feature_importance_by_model'] as Map? ?? {}),
      cvTrainRocAucMean:
          (json['cv_train_roc_auc_mean'] as num?)?.toDouble() ?? 0.0,
      cvTrainRocAucStd:
          (json['cv_train_roc_auc_std'] as num?)?.toDouble() ?? 0.0,
      generatedAt: json['generated_at']?.toString() ?? '',
      charts: (json['charts'] as List<dynamic>? ?? [])
          .map((e) => ChartAsset.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
    );
  }
}

class ChartAsset {
  ChartAsset({
    required this.title,
    required this.assetPath,
    required this.category,
  });

  final String title;
  final String assetPath;
  final String category;

  factory ChartAsset.fromJson(Map<String, dynamic> json) {
    return ChartAsset(
      title: json['title']?.toString() ?? 'Grafik',
      assetPath: json['asset_path']?.toString() ?? '',
      category: json['category']?.toString() ?? 'analysis',
    );
  }
}
