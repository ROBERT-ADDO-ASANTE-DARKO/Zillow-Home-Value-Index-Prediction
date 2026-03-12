import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';

/// Line chart for the composite AHPI historical time series.
class AhpiChart extends StatelessWidget {
  const AhpiChart({super.key, required this.data});

  final List<AhpiPoint> data;

  static const _lineColor = Color(0xFF4CAF50);
  static const _gridColor = Color(0xFF1E2A1E);

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) {
      return const Center(
        child: Text('No data', style: TextStyle(color: Colors.grey)),
      );
    }

    final spots = data
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value.value))
        .toList();

    final minY = data.map((p) => p.value).reduce((a, b) => a < b ? a : b) * 0.95;
    final maxY = data.map((p) => p.value).reduce((a, b) => a > b ? a : b) * 1.05;

    return LineChart(
      LineChartData(
        minY: minY,
        maxY: maxY,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => FlLine(
            color: _gridColor,
            strokeWidth: 1,
          ),
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (value, _) => Text(
                value.toStringAsFixed(0),
                style: const TextStyle(color: Colors.grey, fontSize: 10),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              interval: (data.length / 6).ceilToDouble(),
              getTitlesWidget: (value, _) {
                final idx = value.toInt();
                if (idx < 0 || idx >= data.length) return const SizedBox.shrink();
                // Show year only for readability
                final year = data[idx].date.split('-')[0];
                return Text(
                  year,
                  style: const TextStyle(color: Colors.grey, fontSize: 10),
                );
              },
            ),
          ),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.35,
            color: _lineColor,
            barWidth: 2.5,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: _lineColor.withOpacity(0.12),
            ),
          ),
        ],
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (_) => const Color(0xFF1E2A1E),
            getTooltipItems: (spots) => spots
                .map(
                  (s) => LineTooltipItem(
                    '${data[s.x.toInt()].date}\n${s.y.toStringAsFixed(1)}',
                    const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                )
                .toList(),
          ),
        ),
      ),
    );
  }
}
