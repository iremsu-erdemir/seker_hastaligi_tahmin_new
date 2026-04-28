import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/design_system.dart';

class ResultGauge extends StatelessWidget {
  const ResultGauge({
    super.key,
    required this.risk,
    required this.color,
  });

  final double risk;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: risk.clamp(0.0, 1.0)),
      duration: const Duration(milliseconds: 900),
      curve: Curves.easeOutCubic,
      builder: (context, value, _) {
        return SizedBox(
          height: AppSpacing.xl * 5,
          child: CustomPaint(
            painter: _GaugePainter(progress: value, color: color),
            child: Center(
              child: Text(
                '%${(value * 100).toStringAsFixed(0)}',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter({required this.progress, required this.color});

  final double progress;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.9);
    final radius = math.min(size.width * 0.42, size.height * 0.7);
    const startAngle = math.pi;
    const sweepAngle = math.pi;

    final background =
        Paint()
          ..color = AppColors.border
          ..style = PaintingStyle.stroke
          ..strokeWidth = AppSpacing.sm + AppSpacing.xs + 2
          ..strokeCap = StrokeCap.round;

    final foreground =
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = AppSpacing.sm + AppSpacing.xs + 2
          ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      background,
    );

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle * progress,
      false,
      foreground,
    );
  }

  @override
  bool shouldRepaint(covariant _GaugePainter oldDelegate) {
    return oldDelegate.progress != progress || oldDelegate.color != color;
  }
}
