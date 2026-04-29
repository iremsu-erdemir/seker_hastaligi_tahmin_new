// API katmanında kullanılan veri modelleri.

class PredictResult {
  PredictResult({
    required this.prediction,
    required this.riskScore,
    required this.riskCategory,
    required this.risk,
    required this.classId,
    required this.threshold,
    required this.modelInfo,
    required this.explanations,
    required this.topContributors,
    required this.modelHealth,
    required this.driftStatus,
    required this.riskDistribution,
    required this.inferenceId,
  });

  final int prediction;
  final double riskScore;
  final String riskCategory;
  final double risk;
  final int classId;
  final double threshold;
  final ModelInfo modelInfo;
  final List<String> explanations;
  final List<Map<String, dynamic>> topContributors;
  final ModelHealth modelHealth;
  final String driftStatus;
  final RiskDistribution riskDistribution;
  final String? inferenceId;

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    return PredictResult(
      prediction: (json['prediction'] as num?)?.toInt() ??
          (json['class'] as num?)?.toInt() ??
          0,
      riskScore: (json['risk_score'] as num?)?.toDouble() ??
          (json['risk'] as num?)?.toDouble() ??
          0.0,
      riskCategory: json['risk_category']?.toString() ?? '-',
      risk: (json['risk'] as num?)?.toDouble() ?? 0.0,
      classId: (json['class'] as num?)?.toInt() ?? 0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      modelInfo: ModelInfo.fromJson(
          Map<String, dynamic>.from(json['model_info'] as Map? ?? {})),
      explanations: (json['explanation'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      topContributors: (json['top_contributors'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList(),
      modelHealth: ModelHealth.fromJson(
          Map<String, dynamic>.from(json['model_health'] as Map? ?? {})),
      driftStatus: json['drift_status']?.toString() ?? 'OK',
      riskDistribution: RiskDistribution.fromJson(
          Map<String, dynamic>.from(json['risk_distribution'] as Map? ?? {})),
      inferenceId: json['inference_id']?.toString(),
    );
  }
}

class ModelInfo {
  ModelInfo({
    required this.rocAuc,
    required this.recall,
    required this.threshold,
  });

  final double rocAuc;
  final double recall;
  final double threshold;

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      rocAuc: (json['roc_auc'] as num?)?.toDouble() ?? 0.0,
      recall: (json['recall'] as num?)?.toDouble() ?? 0.0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
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
    required this.classificationReportDict,
    required this.confusionMatrix,
    required this.featureImportanceByModel,
    required this.cvTrainRocAucMean,
    required this.cvTrainRocAucStd,
    required this.generatedAt,
    required this.charts,
    required this.clinicalOptimization,
    required this.calibration,
    required this.monitoring,
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
  final Map<String, dynamic> classificationReportDict;
  final List<List<int>> confusionMatrix;
  final Map<String, dynamic> featureImportanceByModel;
  final double cvTrainRocAucMean;
  final double cvTrainRocAucStd;
  final String generatedAt;
  final List<ChartAsset> charts;
  final Map<String, dynamic> clinicalOptimization;
  final Map<String, dynamic> calibration;
  final Map<String, dynamic> monitoring;

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
      classificationReportDict: Map<String, dynamic>.from(
          json['classification_report_test_best_dict'] as Map? ?? {}),
      confusionMatrix: (json['confusion_matrix'] as List<dynamic>? ?? [])
          .map((row) => (row as List<dynamic>)
              .map((v) => (v as num?)?.toInt() ?? 0)
              .toList())
          .toList(),
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
      clinicalOptimization:
          Map<String, dynamic>.from(json['clinical_optimization'] as Map? ?? {}),
      calibration: Map<String, dynamic>.from(json['calibration'] as Map? ?? {}),
      monitoring: Map<String, dynamic>.from(json['monitoring'] as Map? ?? {}),
    );
  }
}

class ModelHealth {
  ModelHealth({
    required this.rocAuc,
    required this.recall,
    required this.brierScore,
    required this.predictionDistributionShift,
    required this.hasLabeledMetrics,
    required this.labeledSampleCount,
    required this.minLabeledForMetrics,
    required this.classesSeen,
    required this.recallStatus,
  });

  final double rocAuc;
  final double? recall;
  final double brierScore;
  final double predictionDistributionShift;
  final bool hasLabeledMetrics;
  final int labeledSampleCount;
  final int minLabeledForMetrics;
  final List<int> classesSeen;
  final String recallStatus;

  factory ModelHealth.fromJson(Map<String, dynamic> json) {
    return ModelHealth(
      rocAuc: (json['roc_auc'] as num?)?.toDouble() ?? 0.0,
      recall: (json['recall'] as num?)?.toDouble(),
      brierScore: (json['brier_score'] as num?)?.toDouble() ?? 0.0,
      predictionDistributionShift:
          (json['prediction_distribution_shift'] as num?)?.toDouble() ?? 0.0,
      hasLabeledMetrics: json['has_labeled_metrics'] == true,
      labeledSampleCount: (json['labeled_sample_count'] as num?)?.toInt() ?? 0,
      minLabeledForMetrics:
          (json['min_labeled_for_metrics'] as num?)?.toInt() ?? 20,
      classesSeen: (json['classes_seen'] as List<dynamic>? ?? [])
          .map((e) => (e as num).toInt())
          .toList(),
      recallStatus: json['recall_status']?.toString() ?? 'collecting_samples',
    );
  }
}

class RiskDistribution {
  RiskDistribution({
    required this.lowRiskRatio,
    required this.mediumRiskRatio,
    required this.highRiskRatio,
  });

  final double lowRiskRatio;
  final double mediumRiskRatio;
  final double highRiskRatio;

  factory RiskDistribution.fromJson(Map<String, dynamic> json) {
    return RiskDistribution(
      lowRiskRatio: (json['low_risk_ratio'] as num?)?.toDouble() ?? 0.0,
      mediumRiskRatio: (json['medium_risk_ratio'] as num?)?.toDouble() ?? 0.0,
      highRiskRatio: (json['high_risk_ratio'] as num?)?.toDouble() ?? 0.0,
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
