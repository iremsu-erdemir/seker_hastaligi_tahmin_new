import 'package:flutter/material.dart';

import '../core/design_system.dart';
import 'result_gauge.dart';

class RiskResultCard extends StatelessWidget {
  const RiskResultCard({
    super.key,
    required this.risk,
    required this.riskLabel,
    required this.riskMessage,
    required this.color,
  });

  final double risk;
  final String riskLabel;
  final String riskMessage;
  final Color color;
  final bool isVisible = true;

  @override
  Widget build(BuildContext context) {
    return AnimatedScale(
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutBack,
      scale: isVisible ? 1 : 0.96,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 320),
        opacity: isVisible ? 1 : 0,
        child: DsCard(
          padding: const EdgeInsets.all(AppSpacing.lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Risk Sonucu', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: AppSpacing.md),
              ResultGauge(risk: risk, color: color),
              const SizedBox(height: AppSpacing.md),
              Text(
                'Risk: %${(risk * 100).toStringAsFixed(1)} ($riskLabel)',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(color: color),
              ),
              const SizedBox(height: AppSpacing.sm),
              Text(
                riskMessage,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
