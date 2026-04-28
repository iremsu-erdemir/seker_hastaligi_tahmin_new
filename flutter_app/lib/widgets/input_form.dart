import 'package:flutter/material.dart';

import '../core/design_system.dart';

class PredictionInputForm extends StatelessWidget {
  const PredictionInputForm({
    super.key,
    required this.glucoseController,
    required this.bmiController,
    required this.ageController,
    required this.bloodPressureController,
    required this.insulinController,
    required this.onSubmit,
    required this.isLoading,
  });

  final TextEditingController glucoseController;
  final TextEditingController bmiController;
  final TextEditingController ageController;
  final TextEditingController bloodPressureController;
  final TextEditingController insulinController;
  final VoidCallback onSubmit;
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    return DsCard(
      padding: const EdgeInsets.all(AppSpacing.lg),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Hasta Verileri', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: AppSpacing.sm),
          Text(
            'Lütfen temel değerleri giriniz.',
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: AppSpacing.md),
          _numberField(label: 'Glucose', controller: glucoseController),
          const SizedBox(height: AppSpacing.md),
          _numberField(label: 'BMI', controller: bmiController),
          const SizedBox(height: AppSpacing.md),
          _numberField(label: 'Age', controller: ageController),
          const SizedBox(height: AppSpacing.md),
          _numberField(
            label: 'Blood Pressure',
            controller: bloodPressureController,
          ),
          const SizedBox(height: AppSpacing.md),
          _numberField(label: 'Insulin', controller: insulinController),
          const SizedBox(height: AppSpacing.lg),
          DsPrimaryButton(
            onPressed: onSubmit,
            enabled: !isLoading,
            label: isLoading ? 'Risk hesaplanıyor...' : 'Riski Hesapla',
          ),
        ],
      ),
    );
  }

  Widget _numberField({
    required String label,
    required TextEditingController controller,
  }) {
    return TextField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        labelText: label,
      ),
    );
  }
}
