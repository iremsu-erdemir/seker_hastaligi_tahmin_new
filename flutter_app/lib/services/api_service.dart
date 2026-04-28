import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/api_models.dart';

// Python backend ile haberleşen servis sınıfı.
class DiabetesApiService {
  DiabetesApiService({String? baseUrl})
      : _baseUrl = baseUrl ?? _defaultBaseUrl();

  final String _baseUrl;

  static String _defaultBaseUrl() {
    const fromEnv = String.fromEnvironment('API_BASE_URL');
    if (fromEnv.isNotEmpty) {
      return fromEnv;
    }
    // Android emulator'da host makineye 10.0.2.2 ile erişilir.
    // Flutter web/desktop tarafında localhost kullanılır.
    return kIsWeb ? 'http://127.0.0.1:8000' : 'http://10.0.2.2:8000';
  }

  // /predict endpoint'ine formdan gelen veriyi gönderir.
  Future<PredictResult> predict({
    required double glucose,
    required double bmi,
    required double age,
    required double bloodPressure,
    required double insulin,
  }) async {
    final uri = Uri.parse('$_baseUrl/predict');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'Glucose': glucose,
        'BMI': bmi,
        'Age': age,
        'BloodPressure': bloodPressure,
        'Insulin': insulin,
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Tahmin servisine ulaşılamadı: ${response.statusCode}');
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return PredictResult.fromJson(decoded);
  }

  // /metrics endpoint'inden model performans metriklerini alır.
  Future<MetricsResult> fetchMetrics() async {
    final uri = Uri.parse('$_baseUrl/metrics');
    final response = await http.get(uri);
    if (response.statusCode != 200) {
      throw Exception('Metrik servisine ulaşılamadı: ${response.statusCode}');
    }
    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return MetricsResult.fromJson(decoded);
  }
}

