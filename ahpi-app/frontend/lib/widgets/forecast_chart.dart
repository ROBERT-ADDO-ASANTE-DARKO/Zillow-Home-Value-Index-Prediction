import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../models/models.dart';

/// Overlays historical AHPI with a forecast band (yhat ± CI).
class ForecastChart extends StatelessWidget {
  const ForecastChart({
    super.key,
    required this.historical,
    required this.forecast,
    required this.scenario,
  });

  final List<AhpiPoint> historical;
  final List<ForecastPoint> forecast;
  final String scenario;

  static final _scenarioColors = {
    'bear': const Color(0xFFE57373),
    'base': const Color(0xFF4CAF50),
    'bull': const Color(0xFF64B5F6),
  };

  @override
  Widget build(BuildContext context) {
    if (historical.isEmpty && forecast.isEmpty) {
      return const Center(child: Text('No data', style: TextStyle(color: Colors.grey)));
    }

    final lineColor = _scenarioColors[scenario] ?? const Color(0xFF4CAF50);

    // Combine dates for a shared x-axis (index-based)
    final allDates = [
      ...historical.map((p) => p.date),
      ...forecast.map((p) => p.date),
    ];
    final histSpots = historical
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value.value))
        .toList();

    final histLen = historical.length;
    final forecastSpots = forecast
        .asMap()
        .entries
        .map((e) => FlSpot((histLen + e.key).toDouble(), e.value.yhat))
        .toList();

    final upperBand = forecast
        .asMap()
        .entries
        .map((e) => FlSpot((histLen + e.key).toDouble(), e.value.yhatUpper))
        .toList();
    final lowerBand = forecast
        .asMap()
        .entries
        .map((e) => FlSpot((histLen + e.key).toDouble(), e.value.yhatLower))
        .toList();

    final allValues = [
      ...historical.map((p) => p.value),
      ...forecast.map((p) => p.yhatUpper),
      ...forecast.map((p) => p.yhatLower),
    ];
    final minY = allValues.reduce((a, b) => a < b ? a : b) * 0.95;
    final maxY = allValues.reduce((a, b) => a > b ? a : b) * 1.05;

    return LineChart(
      LineChartData(
        minY: minY,
        maxY: maxY,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) =>
              const FlLine(color: Color(0xFF1E2A1E), strokeWidth: 1),
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (v, _) => Text(
                v.toStringAsFixed(0),
                style: const TextStyle(color: Colors.grey, fontSize: 10),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              interval: (allDates.length / 6).ceilToDouble(),
              getTitlesWidget: (v, _) {
                final idx = v.toInt();
                if (idx < 0 || idx >= allDates.length) return const SizedBox.shrink();
                return Text(
                  allDates[idx].split('-')[0],
                  style: const TextStyle(color: Colors.grey, fontSize: 10),
                );
              },
            ),
          ),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          // Historical line
          LineChartBarData(
            spots: histSpots,
            isCurved: true,
            color: Colors.white70,
            barWidth: 2,
            dotData: const FlDotData(show: false),
          ),
          // Forecast line
          LineChartBarData(
            spots: forecastSpots,
            isCurved: true,
            color: lineColor,
            barWidth: 2.5,
            dotData: const FlDotData(show: false),
            dashArray: [6, 3],
          ),
          // Upper CI band (transparent fill trick via belowBarData of upper - lower)
          LineChartBarData(
            spots: upperBand,
            isCurved: true,
            color: Colors.transparent,
            barWidth: 0,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: lineColor.withOpacity(0.15),
              spotsLine: BarAreaSpotsLine(show: false),
            ),
          ),
          LineChartBarData(
            spots: lowerBand,
            isCurved: true,
            color: Colors.transparent,
            barWidth: 0,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: const Color(0xFF0D1117),
              spotsLine: BarAreaSpotsLine(show: false),
            ),
          ),
        ],
      ),
    );
  }
}
