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

    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Diyabet Karar Destek Sistemi'),
          bottom: const TabBar(
            isScrollable: true,
            tabs: [
              Tab(text: 'Tahmin', icon: Icon(Icons.monitor_heart_outlined)),
              Tab(text: 'Açıklama', icon: Icon(Icons.lightbulb_outline)),
              Tab(text: 'Performans', icon: Icon(Icons.analytics_outlined)),
              Tab(text: 'Analiz', icon: Icon(Icons.insights_outlined)),
            ],
          ),
        ),
        body: TabBarView(
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
                            padding:
                                const EdgeInsets.only(bottom: AppSpacing.md),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(name,
                                    style:
                                        Theme.of(context).textTheme.bodyLarge),
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
                      metrics != null
                          ? metrics.accuracy.toStringAsFixed(3)
                          : '-',
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
                      'Eşik Değeri',
                      metrics != null
                          ? metrics.threshold.toStringAsFixed(3)
                          : '-',
                    ),
                  ],
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard('ROC Eğrisi', 'assets/charts/roc.png'),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                    'Karışıklık Matrisi', 'assets/charts/confusion_matrix.png'),
              ],
            ),
            ListView(
              padding: const EdgeInsets.all(AppSpacing.lg),
              children: [
                _sectionHeader(
                  'Analiz Ekrani',
                  'Bu ekran ikincil bilgi amaçlıdır.',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'Özellik Dağılımı',
                  'assets/charts/eda_glucose_hist.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'Korelasyon Isı Haritası',
                  'assets/charts/eda_correlation_heatmap.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'Sonuç Dağılımı',
                  'assets/charts/eda_outcome_distribution.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'BMI Dağılımı',
                  'assets/charts/eda_bmi_hist.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'Yaş Dağılımı',
                  'assets/charts/eda_age_hist.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                  'Insulin Karşılaştırması',
                  'assets/charts/eda_insulin_winsor_compare.png',
                ),
                const SizedBox(height: AppSpacing.md),
                _chartCard(
                    'Öğrenme Eğrisi', 'assets/charts/eda_learning_curve.png'),
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
                      ],
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
