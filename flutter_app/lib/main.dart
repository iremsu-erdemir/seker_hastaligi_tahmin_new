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

  @override
  void initState() {
    super.initState();
    _loadMetricsFromApi();
  }

  @override
  void dispose() {
    _glucoseController.dispose();
    _bmiController.dispose();
    _ageController.dispose();
    _bpController.dispose();
    _insulinController.dispose();
    super.dispose();
  }

  Future<void> _loadMetricsFromApi() async {
    setState(() {
      _isLoadingMetrics = true;
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

  Future<void> _calculateRisk() async {
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
      _isPredicting = true;
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
    if (risk < 0.4) return AppColors.success;
    if (risk <= 0.6) return AppColors.warning;
    return AppColors.danger;
  }

  String _riskLabel(double risk) {
    if (risk < 0.4) return 'Düşük';
    if (risk <= 0.6) return 'Sınırda';
    return 'Yüksek';
  }

  String _riskMessage(double risk) {
    if (risk < 0.4) return 'Diyabet riski düşük seviyede';
    if (risk <= 0.6) return 'Diyabet riski orta seviyede';
    return 'Diyabet riski yüksek seviyede';
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

  Widget _chartCard(String title, String assetPath) {
    return DsCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: AppSpacing.md),
          ClipRRect(
            borderRadius: BorderRadius.circular(AppRadius.md),
            child: Image.asset(
              assetPath,
              fit: BoxFit.contain,
              errorBuilder: (context, _, __) => const Padding(
                padding: EdgeInsets.all(AppSpacing.md),
                child: Text('Görsel bulunamadı'),
              ),
            ),
          ),
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

  @override
  Widget build(BuildContext context) {
    final metrics = _metrics;
    final predictResult = _predictResult;
    final risk = predictResult?.risk ?? 0;
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
                onSubmit: _calculateRisk,
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
                    riskLabel: _riskLabel(risk),
                    riskMessage: _riskMessage(risk),
                    color: _riskColor(risk),
                  ),
                ),
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
                    'Model en çok Glucose değerine bakarak karar verdi.',
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
                  action: _loadMetricsFromApi,
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
              if (metrics != null)
                DsCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Sınıflandırma Raporu',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      SelectableText(
                        metrics.classificationReport.isEmpty
                            ? 'Rapor bulunamadı'
                            : metrics.classificationReport,
                      ),
                    ],
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
                  child: _chartCard(chart.title, chart.assetPath),
                ),
              ),
              ..._chartsByCategory('model').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(chart.title, chart.assetPath),
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
              ..._chartsByCategory('eda').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(chart.title, chart.assetPath),
                ),
              ),
              ..._chartsByCategory('analysis').map(
                (chart) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.md),
                  child: _chartCard(chart.title, chart.assetPath),
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
