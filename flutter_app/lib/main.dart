import 'dart:async';

import 'package:flutter/material.dart';

import 'core/design_system.dart';
import 'models/api_models.dart';
import 'services/api_service.dart';
import 'widgets/input_form.dart';
import 'widgets/risk_card.dart';

void main() {
  runApp(const DiabetesApp());
}

class DiabetesApp extends StatelessWidget {
  const DiabetesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Diyabet Risk Desteği',
      theme: AppTheme.light,
      debugShowCheckedModeBanner: false,
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _api = DiabetesApiService();
  static const Duration _autoRefreshInterval = Duration(seconds: 8);

  final _glucoseController = TextEditingController(text: '120');
  final _bmiController = TextEditingController(text: '30');
  final _ageController = TextEditingController(text: '45');
  final _bpController = TextEditingController(text: '70');
  final _insulinController = TextEditingController(text: '80');

  PredictResult? _predictResult;
  MetricsResult? _metrics;
  String? _predictError;
  String? _metricsError;
  bool _isPredicting = false;
  bool _isLoadingMetrics = false;
  int _currentIndex = 0;
  int _refreshSeed = DateTime.now().millisecondsSinceEpoch;
  Timer? _autoRefreshTimer;

  @override
  void initState() {
    super.initState();
    _refreshAll(includePrediction: false);
    _startAutoRefresh();
  }

  @override
  void dispose() {
    _autoRefreshTimer?.cancel();
    _glucoseController.dispose();
    _bmiController.dispose();
    _ageController.dispose();
    _bpController.dispose();
    _insulinController.dispose();
    super.dispose();
  }

  Future<void> _loadMetricsFromApiInternal({required bool showLoading}) async {
    setState(() {
      _isLoadingMetrics = showLoading;
      _metricsError = null;
    });
    try {
      final result = await _api.fetchMetrics();
      setState(() {
        _metrics = result;
      });
    } catch (e) {
      setState(() {
        _metricsError = 'Sunucuya ulaşılamadı';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingMetrics = false;
        });
      }
    }
  }

  void _startAutoRefresh() {
    _autoRefreshTimer?.cancel();
    _autoRefreshTimer = Timer.periodic(_autoRefreshInterval, (_) {
      _refreshAll(includePrediction: _predictResult != null, showLoading: false);
    });
  }

  Future<void> _refreshAll({
    bool includePrediction = false,
    bool showLoading = true,
  }) async {
    if (showLoading) {
      setState(() {
        _refreshSeed = DateTime.now().millisecondsSinceEpoch;
      });
    }
    await _loadMetricsFromApiInternal(showLoading: showLoading);
    if (includePrediction) {
      await _calculateRisk(showLoading: showLoading);
    }
  }

  Future<void> _calculateRisk({bool showLoading = true}) async {
    final glucose = double.tryParse(_glucoseController.text.trim());
    final bmi = double.tryParse(_bmiController.text.trim());
    final age = double.tryParse(_ageController.text.trim());
    final bloodPressure = double.tryParse(_bpController.text.trim());
    final insulin = double.tryParse(_insulinController.text.trim());

    if ([glucose, bmi, age, bloodPressure, insulin].any((v) => v == null)) {
      setState(() {
        _predictError = 'Lütfen değerleri giriniz';
        _predictResult = null;
      });
      return;
    }

    setState(() {
      _isPredicting = showLoading;
      _predictError = null;
    });
    try {
      final result = await _api.predict(
        glucose: glucose!,
        bmi: bmi!,
        age: age!,
        bloodPressure: bloodPressure!,
        insulin: insulin!,
      );
      setState(() {
        _predictResult = result;
      });
    } catch (e) {
      setState(() {
        _predictError = 'Sunucuya ulaşılamadı';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isPredicting = false;
        });
      }
    }
  }

  Color _riskColor(double risk) {
    if (risk < 0.3) return AppColors.success;
    if (risk < 0.6) return AppColors.warning;
    return AppColors.danger;
  }

  String _riskLabel(double risk) {
    if (risk < 0.3) return 'Düşük Risk';
    if (risk < 0.6) return 'Orta Risk';
    return 'Yüksek Risk';
  }

  String _riskMessage(double risk) {
    if (risk >= 0.6) return 'Model, diyabet açısından yüksek risk tespit etti.';
    if (risk >= 0.3) return 'Model, diyabet açısından orta risk tespit etti.';
    return 'Model, diyabet açısından düşük risk tespit etti.';
  }

  Widget _metricCard(String title, String value) {
    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.bodyLarge),
          const SizedBox(height: AppSpacing.sm),
          Text(value, style: Theme.of(context).textTheme.titleLarge),
        ],
      ),
    );
  }

  String _chartUrl(String chartPath) {
    final fileName = chartPath.split('/').last;
    return '${_api.baseUrl}/charts/$fileName?t=$_refreshSeed';
  }

  String _formatTurkeyDateTime(String rawDateTime) {
    final parsed = DateTime.tryParse(rawDateTime);
    if (parsed == null) return rawDateTime;

    final turkeyTime = (parsed.isUtc ? parsed : parsed.toUtc())
        .add(const Duration(hours: 3));
    String twoDigits(int value) => value.toString().padLeft(2, '0');

    return '${twoDigits(turkeyTime.day)}.${twoDigits(turkeyTime.month)}.${turkeyTime.year} '
        '${twoDigits(turkeyTime.hour)}:${twoDigits(turkeyTime.minute)}';
  }

  String _chartLiveSummary(String chartPath, MetricsResult? metrics) {
    if (metrics == null) return '';
    final lower = chartPath.toLowerCase();
    if (lower.contains('roc')) {
      return 'Canlı ROC-AUC: ${metrics.rocAuc.toStringAsFixed(3)} | Canlı Eşik: ${metrics.threshold.toStringAsFixed(3)}';
    }
    if (lower.contains('pr')) {
      return 'Canlı Precision(Macro): ${metrics.precisionMacro.toStringAsFixed(3)} | Canlı Recall(Macro): ${metrics.recallMacro.toStringAsFixed(3)}';
    }
    return 'Son güncelleme: ${_formatTurkeyDateTime(metrics.generatedAt)} (TR)';
  }

  Widget _chartCard(String title, String chartPath, {MetricsResult? metrics}) {
    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          ClipRRect(
            borderRadius: BorderRadius.circular(AppRadius.md),
            child: Image.network(
              _chartUrl(chartPath),
              fit: BoxFit.contain,
              gaplessPlayback: true,
              errorBuilder: (context, _, __) => const Padding(
                padding: EdgeInsets.all(AppSpacing.md),
                child: Text('Canlı görsel alınamadı'),
              ),
            ),
          ),
          if (_chartLiveSummary(chartPath, metrics).isNotEmpty) ...[
            const SizedBox(height: AppSpacing.sm),
            Text(
              _chartLiveSummary(chartPath, metrics),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ],
      ),
    );
  }

  List<ChartAsset> _chartsByCategory(String category) {
    final charts = _metrics?.charts ?? <ChartAsset>[];
    return charts.where((c) => c.category == category).toList();
  }

  Widget _modelMetricCard(Map<String, dynamic> model) {
    String asText(String key) {
      final value = model[key];
      if (value == null) return '-';
      if (value is num) return value.toStringAsFixed(3);
      return value.toString();
    }

    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            model['name']?.toString() ?? 'Model',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: AppSpacing.md),
          Text('Accuracy: ${asText('test_accuracy')}'),
          Text('Balanced Accuracy: ${asText('test_balanced_accuracy')}'),
          Text('Precision (Macro): ${asText('test_precision_macro')}'),
          Text('Recall (Macro): ${asText('test_recall_macro')}'),
          Text('F1 (Macro): ${asText('test_f1_macro')}'),
          Text('ROC-AUC: ${asText('test_roc_auc')}'),
        ],
      ),
    );
  }

  double _metricAsDouble(Map<String, dynamic> row, String key) {
    final value = row[key];
    return (value as num?)?.toDouble() ?? 0.0;
  }

  Color _metricHeatColor(double value) {
    if (value >= 0.80) return Colors.green.shade600;
    if (value >= 0.60) return Colors.amber.shade600;
    return Colors.deepOrange.shade400;
  }

  Widget _summaryMetricCards(MetricsResult metrics) {
    Widget card(String title, String value, Color color) {
      return Expanded(
        child: DsCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: AppSpacing.sm),
              Text(
                value,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      color: color,
                      fontWeight: FontWeight.w700,
                    ),
              ),
            ],
          ),
        ),
      );
    }

    return Row(
      children: [
        card('Accuracy', '%${(metrics.accuracy * 100).toStringAsFixed(1)}',
            _metricHeatColor(metrics.accuracy)),
        const SizedBox(width: AppSpacing.md),
        card('Macro Recall', metrics.recallMacro.toStringAsFixed(3),
            _metricHeatColor(metrics.recallMacro)),
        const SizedBox(width: AppSpacing.md),
        card('Macro F1', metrics.f1.toStringAsFixed(3), _metricHeatColor(metrics.f1)),
      ],
    );
  }

  Widget _classificationHeatTable(MetricsResult metrics) {
    final report = metrics.classificationReportDict;
    final class0 = Map<String, dynamic>.from(report['0'] as Map? ?? {});
    final class1 = Map<String, dynamic>.from(report['1'] as Map? ?? {});
    final macro = Map<String, dynamic>.from(report['macro avg'] as Map? ?? {});
    final weighted = Map<String, dynamic>.from(report['weighted avg'] as Map? ?? {});
    String indicator(double value) {
      if (value >= 0.80) return '🟢';
      if (value >= 0.60) return '🟡';
      return '🔴';
    }

    String metricWithIndicator(double value) =>
        '${indicator(value)} ${value.toStringAsFixed(2)}';

    TableCell headerCell(String text, {TextAlign align = TextAlign.left}) {
      return TableCell(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Text(
            text,
            textAlign: align,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
        ),
      );
    }

    TableRow classRow({
      required String label,
      required Map<String, dynamic> row,
    }) {
      final precision = _metricAsDouble(row, 'precision');
      final recall = _metricAsDouble(row, 'recall');
      final f1 = _metricAsDouble(row, 'f1-score');
      final support = _metricAsDouble(row, 'support').toInt();

      return TableRow(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(label),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              metricWithIndicator(precision),
              textAlign: TextAlign.center,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              metricWithIndicator(recall),
              textAlign: TextAlign.center,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              metricWithIndicator(f1),
              textAlign: TextAlign.center,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text('$support', textAlign: TextAlign.right),
          ),
        ],
      );
    }

    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Sınıflandırma Tablosu',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.sm),
          Table(
            columnWidths: const {
              0: FlexColumnWidth(2.1),
              1: FlexColumnWidth(1),
              2: FlexColumnWidth(1),
              3: FlexColumnWidth(1),
              4: FlexColumnWidth(0.8),
            },
            defaultVerticalAlignment: TableCellVerticalAlignment.middle,
            children: [
              TableRow(
                children: [
                  headerCell('Sınıf'),
                  headerCell('Precision', align: TextAlign.center),
                  headerCell('Recall', align: TextAlign.center),
                  headerCell('F1 Score', align: TextAlign.center),
                  headerCell('Support', align: TextAlign.right),
                ],
              ),
              classRow(label: 'Sağlıklı (0)', row: class0),
              classRow(label: 'Diyabet (1)', row: class1),
            ],
          ),
          const SizedBox(height: AppSpacing.md),
          Text(
            'Genel Performans',
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
          const SizedBox(height: AppSpacing.sm),
          Text('Macro Avg F1: ${_metricAsDouble(macro, 'f1-score').toStringAsFixed(2)}'),
          Text(
              'Weighted Avg F1: ${_metricAsDouble(weighted, 'f1-score').toStringAsFixed(2)}'),
          const SizedBox(height: AppSpacing.md),
          Text(
            'Model, diyabet sınıfında yüksek recall (${_metricAsDouble(class1, 'recall').toStringAsFixed(2)}) elde ederek hastaları kaçırmamayı başarıyor ancak düşük precision (${_metricAsDouble(class1, 'precision').toStringAsFixed(2)}) nedeniyle yanlış pozitif oranı yüksektir.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ],
      ),
    );
  }

  Widget _classF1Bars(MetricsResult metrics) {
    final report = metrics.classificationReportDict;
    final class0 = _metricAsDouble(
      Map<String, dynamic>.from(report['0'] as Map? ?? {}),
      'f1-score',
    );
    final class1 = _metricAsDouble(
      Map<String, dynamic>.from(report['1'] as Map? ?? {}),
      'f1-score',
    );

    Widget bar(String label, double value, Color color) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$label  ${value.toStringAsFixed(3)}'),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: value.clamp(0.0, 1.0),
              minHeight: 14,
              color: color,
              backgroundColor: AppColors.border,
            ),
          ),
        ],
      );
    }

    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Sınıf Bazlı F1 Karşılaştırma',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          bar('Sağlıklı (0)', class0, AppColors.primary),
          const SizedBox(height: AppSpacing.md),
          bar('Diyabet (1)', class1, AppColors.danger),
        ],
      ),
    );
  }

  Widget _confusionHeatmapCard(MetricsResult metrics) {
    final cm = metrics.confusionMatrix;
    if (cm.length < 2 || cm[0].length < 2 || cm[1].length < 2) {
      return const SizedBox.shrink();
    }
    final values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]];
    final maxV = values.reduce((a, b) => a > b ? a : b).toDouble();
    Color cellColor(int v) {
      final t = maxV <= 0 ? 0.0 : v / maxV;
      return Color.lerp(Colors.orange.shade100, Colors.green.shade600, t)!;
    }

    Widget cell(int v) => Container(
          height: 72,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: cellColor(v),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            '$v',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
        );

    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Confusion Matrix',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          const Text('Pred: Sağlıklı        Pred: Diyabet'),
          const SizedBox(height: 8),
          Row(children: [const SizedBox(width: 64), Expanded(child: cell(cm[0][0])), const SizedBox(width: 8), Expanded(child: cell(cm[0][1]))]),
          const SizedBox(height: 8),
          Row(children: [const SizedBox(width: 64), Expanded(child: cell(cm[1][0])), const SizedBox(width: 8), Expanded(child: cell(cm[1][1]))]),
          const SizedBox(height: 8),
          const Text('True 0 / True 1 satırları temsil eder'),
        ],
      ),
    );
  }

  Widget _classificationDashboard(MetricsResult metrics) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Model Performans Analizi',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: AppSpacing.sm),
        Text('Son güncelleme: ${_formatTurkeyDateTime(metrics.generatedAt)} (TR)',
            style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: AppSpacing.md),
        _summaryMetricCards(metrics),
        const SizedBox(height: AppSpacing.md),
        LayoutBuilder(
          builder: (context, constraints) {
            if (constraints.maxWidth < 900) {
              return Column(
                children: [
                  _classificationHeatTable(metrics),
                  const SizedBox(height: AppSpacing.md),
                  _classF1Bars(metrics),
                  const SizedBox(height: AppSpacing.md),
                  _confusionHeatmapCard(metrics),
                ],
              );
            }
            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 2,
                  child: Column(
                    children: [
                      _classificationHeatTable(metrics),
                      const SizedBox(height: AppSpacing.md),
                      _classF1Bars(metrics),
                    ],
                  ),
                ),
                const SizedBox(width: AppSpacing.md),
                Expanded(child: _confusionHeatmapCard(metrics)),
              ],
            );
          },
        ),
      ],
    );
  }

  List<Map<String, dynamic>> _topContributors() {
    final items = _predictResult?.topContributors ?? <Map<String, dynamic>>[];
    return items.take(3).toList();
  }

  Widget _stateCard({
    required IconData icon,
    required String text,
    required Color color,
    VoidCallback? action,
  }) {
    return DsCard(
      child: Row(
        children: [
          Icon(icon, color: color),
          const SizedBox(width: AppSpacing.md),
          Expanded(
              child: Text(text, style: Theme.of(context).textTheme.bodyLarge)),
          if (action != null)
            IconButton(onPressed: action, icon: const Icon(Icons.refresh)),
        ],
      ),
    );
  }

  Widget _emptyState({required String message, required IconData icon}) {
    return DsCard(
      child: Column(
        children: [
          Icon(icon,
              size: AppSpacing.xl + AppSpacing.lg,
              color: AppColors.textSecondary),
          const SizedBox(height: AppSpacing.md),
          Text(message, style: Theme.of(context).textTheme.bodyLarge),
        ],
      ),
    );
  }

  Widget _sectionHeader(String title, String subtitle) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: AppSpacing.sm),
        Text(subtitle, style: Theme.of(context).textTheme.bodyLarge),
      ],
    );
  }

  Widget _modelHealthPanel(PredictResult result) {
    final health = result.modelHealth;
    final hasLabeled = health.hasLabeledMetrics && health.recall != null;
    final recallText = hasLabeled ? health.recall!.toStringAsFixed(3) : 'Insufficient data';
    String statusHint;
    if (health.recallStatus == 'ready') {
      statusHint = 'Recall metric hazır';
    } else if (health.recallStatus == 'need_both_classes') {
      final classes = health.classesSeen.join(', ');
      statusHint = 'Both classes required. Görülen sınıflar: ${classes.isEmpty ? '-' : classes}';
    } else {
      statusHint =
          'Collecting samples (${health.labeledSampleCount}/${health.minLabeledForMetrics})';
    }
    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Model Health Panel', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          Text('Drift Status: ${result.driftStatus}'),
          Text('ROC-AUC (Latest Batch): ${health.rocAuc.toStringAsFixed(3)}'),
          Text('Recall (Latest Batch): $recallText'),
          Text('Status: $statusHint'),
          Text('Threshold (Active): ${result.threshold.toStringAsFixed(4)}'),
        ],
      ),
    );
  }

  Widget _riskDistributionWidget(PredictResult result) {
    String pct(double value) => '${(value * 100).toStringAsFixed(1)}%';
    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Risk Distribution', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          Text('Düşük Risk: ${pct(result.riskDistribution.lowRiskRatio)}'),
          Text('Orta Risk: ${pct(result.riskDistribution.mediumRiskRatio)}'),
          Text('Yüksek Risk: ${pct(result.riskDistribution.highRiskRatio)}'),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final metrics = _metrics;
    final predictResult = _predictResult;
    final risk = predictResult?.riskScore ?? predictResult?.risk ?? 0;
    final contributors = _topContributors();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diyabet Karar Destek Sistemi'),
      ),
      body: IndexedStack(
        index: _currentIndex,
        children: [
          ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              _sectionHeader(
                'Diyabet Risk Tahmini',
                'Temel parametrelerle risk seviyesini saniyeler içinde görün.',
              ),
              const SizedBox(height: AppSpacing.lg),
              PredictionInputForm(
                glucoseController: _glucoseController,
                bmiController: _bmiController,
                ageController: _ageController,
                bloodPressureController: _bpController,
                insulinController: _insulinController,
                onSubmit: () => _calculateRisk(showLoading: true),
                isLoading: _isPredicting,
              ),
              const SizedBox(height: AppSpacing.md),
              if (_predictError != null)
                _stateCard(
                  icon: Icons.cloud_off,
                  text: _predictError!,
                  color: AppColors.danger,
                ),
              if (predictResult == null &&
                  _predictError == null &&
                  !_isPredicting)
                _emptyState(
                  message: 'Lütfen değerleri giriniz',
                  icon: Icons.info_outline,
                ),
              if (_isPredicting) const DsShimmer(height: AppSpacing.xl * 6),
              if (predictResult != null)
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 320),
                  child: RiskResultCard(
                    key: ValueKey<double>(predictResult.risk),
                    risk: risk,
                    riskLabel: predictResult.riskCategory.isNotEmpty
                        ? predictResult.riskCategory
                        : _riskLabel(risk),
                    riskMessage: _riskMessage(risk),
                    color: _riskColor(risk),
                    rocAuc: predictResult.modelInfo.rocAuc,
                    recall: predictResult.modelInfo.recall,
                    threshold: predictResult.modelInfo.threshold,
                  ),
                ),
              if (predictResult != null) const SizedBox(height: AppSpacing.md),
              if (predictResult != null) _modelHealthPanel(predictResult),
              if (predictResult != null) const SizedBox(height: AppSpacing.md),
              if (predictResult != null) _riskDistributionWidget(predictResult),
            ],
          ),
          ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              _sectionHeader(
                'Bu Sonuca Etki Eden Faktörler',
                'Model kararında en etkili ilk 3 değişken.',
              ),
              const SizedBox(height: AppSpacing.md),
              if (predictResult == null)
                _emptyState(
                  message: 'Önce Tahmin ekranından risk hesaplayınız',
                  icon: Icons.analytics_outlined,
                ),
              if (predictResult != null)
                DsCard(
                  child: Text(
                    contributors.isNotEmpty
                        ? 'Model en çok ${contributors.first['feature'] ?? '-'} değerine bakarak karar verdi.'
                        : 'Canlı açıklama verisi bekleniyor.',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                ),
              const SizedBox(height: AppSpacing.md),
              if (predictResult != null)
                DsCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('En Etkili 3 Özellik',
                          style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: AppSpacing.md),
                      ...contributors.map((item) {
                        final name = item['feature']?.toString() ?? '-';
                        final value =
                            (item['importance'] as num?)?.toDouble() ?? 0.0;
                        final normalized = value.clamp(0.0, 1.0);
                        return Padding(
                          padding: const EdgeInsets.only(bottom: AppSpacing.md),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(name,
                                  style: Theme.of(context).textTheme.bodyLarge),
                              const SizedBox(height: AppSpacing.sm),
                              ClipRRect(
                                borderRadius:
                                    BorderRadius.circular(AppRadius.md),
                                child: LinearProgressIndicator(
                                  value: normalized == 0 ? 0.05 : normalized,
                                  minHeight: AppSpacing.sm + AppSpacing.xs,
                                  color: AppColors.primary,
                                  backgroundColor: AppColors.border,
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                    ],
                  ),
                ),
            ],
          ),
          ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              _sectionHeader(
                'Model Performansı',
                'Jüri değerlendirmesi için temel metrikler ve grafikler.',
              ),
              const SizedBox(height: AppSpacing.md),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.icon(
                  onPressed: _isLoadingMetrics
                      ? null
                      : () =>
                          _refreshAll(includePrediction: _predictResult != null, showLoading: true),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Yenile'),
                ),
              ),
              const SizedBox(height: AppSpacing.md),
              if (_isLoadingMetrics)
                _stateCard(
                  icon: Icons.hourglass_top,
                  text: 'Risk hesaplanıyor...',
                  color: AppColors.primary,
                ),
              if (_metricsError != null)
                _stateCard(
                  icon: Icons.cloud_off,
                  text: _metricsError!,
                  color: AppColors.danger,
                  action: () =>
                      _refreshAll(includePrediction: _predictResult != null, showLoading: true),
                ),
              GridView.count(
                shrinkWrap: true,
                crossAxisCount: 2,
                crossAxisSpacing: AppSpacing.md,
                mainAxisSpacing: AppSpacing.md,
                childAspectRatio: 1.45,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  _metricCard(
                    'Doğruluk',
                    metrics != null ? metrics.accuracy.toStringAsFixed(3) : '-',
                  ),
                  _metricCard(
                    'F1 Skoru',
                    metrics != null ? metrics.f1.toStringAsFixed(3) : '-',
                  ),
                  _metricCard(
                    'ROC-AUC',
                    metrics != null ? metrics.rocAuc.toStringAsFixed(3) : '-',
                  ),
                  _metricCard(
                    'Balanced Accuracy',
                    metrics != null
                        ? metrics.balancedAccuracy.toStringAsFixed(3)
                        : '-',
                  ),
                  _metricCard(
                    'Precision (Macro)',
                    metrics != null
                        ? metrics.precisionMacro.toStringAsFixed(3)
                        : '-',
                  ),
                  _metricCard(
                    'Recall (Macro)',
                    metrics != null
                        ? metrics.recallMacro.toStringAsFixed(3)
                        : '-',
                  ),
                  _metricCard(
                    'Eşik Değeri',
                    metrics != null
                        ? metrics.threshold.toStringAsFixed(3)
                        : '-',
                  ),
                  _metricCard(
                    'CV ROC-AUC Ort.',
                    metrics != null
                        ? metrics.cvTrainRocAucMean.toStringAsFixed(3)
                        : '-',
                  ),
                  _metricCard(
                    'CV ROC-AUC Std',
                    metrics != null
                        ? metrics.cvTrainRocAucStd.toStringAsFixed(3)
                        : '-',
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.lg),
              if (metrics != null) _classificationDashboard(metrics),
              if (metrics != null && metrics.calibration.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: AppSpacing.md),
                  child: DsCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Calibration',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: AppSpacing.sm),
                        Text('Method: ${metrics.calibration['method'] ?? '-'}'),
                        Text(
                          'Brier (Before): ${((metrics.calibration['brier_score_before'] as num?)?.toDouble() ?? 0).toStringAsFixed(4)}',
                        ),
                        Text(
                          'Brier (After): ${((metrics.calibration['brier_score_after'] as num?)?.toDouble() ?? 0).toStringAsFixed(4)}',
                        ),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: AppSpacing.md),
              if (metrics != null)
                ...metrics.models.map((m) => Padding(
                      padding: const EdgeInsets.only(bottom: AppSpacing.md),
                      child: _modelMetricCard(m),
                    )),
              const SizedBox(height: AppSpacing.md),
              ..._chartsByCategory('performance').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(
                    chart.title,
                    chart.assetPath,
                    metrics: metrics,
                  ),
                ),
              ),
              ..._chartsByCategory('model').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(
                    chart.title,
                    chart.assetPath,
                    metrics: metrics,
                  ),
                ),
              ),
            ],
          ),
          ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              _sectionHeader(
                'Analiz ve EDA',
                'Tüm EDA, preprocessing ve ek analiz çıktıları.',
              ),
              const SizedBox(height: AppSpacing.md),
              if (metrics != null)
                DsCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Preprocessing Adımları',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      ...metrics.preprocessing.map(
                        (step) => Padding(
                          padding: const EdgeInsets.only(bottom: AppSpacing.sm),
                          child: Text('• $step'),
                        ),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: AppSpacing.md),
              ..._chartsByCategory('analysis').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(
                    chart.title,
                    chart.assetPath,
                    metrics: metrics,
                  ),
                ),
              ),
              if (metrics != null)
                DsCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Model',
                          style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: AppSpacing.sm),
                      Text(metrics.modelName,
                          style: Theme.of(context).textTheme.bodyLarge),
                      const SizedBox(height: AppSpacing.sm),
                      Text('Üretim zamanı: ${metrics.generatedAt}'),
                    ],
                  ),
                ),
            ],
          ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.monitor_heart_outlined),
            label: 'Tahmin',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.lightbulb_outline),
            label: 'Açıklama',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.analytics_outlined),
            label: 'Performans',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.insights_outlined),
            label: 'Analiz',
          ),
        ],
      ),
    );
  }
}
