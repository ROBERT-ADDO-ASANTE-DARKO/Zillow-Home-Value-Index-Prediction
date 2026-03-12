import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../models/models.dart';

/// Multi-line chart for all district AHPI time series.
class DistrictChart extends StatelessWidget {
  const DistrictChart({super.key, required this.data});

  final List<LocationPoint> data;

  static const _districtColors = {
    'Spintex Road': Color(0xFF64B5F6),
    'Adenta': Color(0xFF81C784),
    'Tema': Color(0xFFFFD54F),
    'Dome': Color(0xFFCE93D8),
    'Kasoa': Color(0xFFFFB74D),
  };

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) {
      return const Center(child: Text('No data', style: TextStyle(color: Colors.grey)));
    }

    // Group by district
    final grouped = <String, List<LocationPoint>>{};
    for (final p in data) {
      grouped.putIfAbsent(p.location, () => []).add(p);
    }
    for (final v in grouped.values) {
      v.sort((a, b) => a.date.compareTo(b.date));
    }

    // Use dates from the first group as shared x-axis
    final dates = grouped.values.first.map((p) => p.date).toList();

    final bars = grouped.entries.map((entry) {
      final color = _districtColors[entry.key] ?? Colors.grey;
      final spots = entry.value
          .asMap()
          .entries
          .map((e) => FlSpot(e.key.toDouble(), e.value.value))
          .toList();
      return LineChartBarData(
        spots: spots,
        isCurved: true,
        curveSmoothness: 0.3,
        color: color,
        barWidth: 2,
        dotData: const FlDotData(show: false),
      );
    }).toList();

    final allValues = data.map((p) => p.value);
    final minY = allValues.reduce((a, b) => a < b ? a : b) * 0.95;
    final maxY = allValues.reduce((a, b) => a > b ? a : b) * 1.05;

    return Column(
      children: [
        Expanded(
          child: LineChart(
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
                    interval: (dates.length / 5).ceilToDouble(),
                    getTitlesWidget: (v, _) {
                      final idx = v.toInt();
                      if (idx < 0 || idx >= dates.length) return const SizedBox.shrink();
                      return Text(
                        dates[idx].split('-')[0],
                        style: const TextStyle(color: Colors.grey, fontSize: 10),
                      );
                    },
                  ),
                ),
                topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              ),
              lineBarsData: bars,
            ),
          ),
        ),
        // Legend
        const SizedBox(height: 8),
        Wrap(
          spacing: 16,
          children: grouped.keys.map((district) {
            final color = _districtColors[district] ?? Colors.grey;
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(width: 12, height: 3, color: color),
                const SizedBox(width: 4),
                Text(district,
                    style: const TextStyle(color: Colors.grey, fontSize: 11)),
              ],
            );
          }).toList(),
        ),
      ],
    );
  }
}
