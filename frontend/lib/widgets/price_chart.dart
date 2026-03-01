import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../models/app_models.dart';
import '../providers/app_provider.dart';

class PriceAnalysisSection extends StatelessWidget {
  const PriceAnalysisSection({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.attach_money, color: Color(0xFF1565C0), size: 22),
                SizedBox(width: 10),
                Text(
                  'Price Analysis',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (provider.dataState == LoadingState.loading)
              const SizedBox(
                height: 280,
                child: Center(child: CircularProgressIndicator()),
              )
            else if (provider.cityHistory.isEmpty)
              const SizedBox(
                height: 200,
                child: Center(
                  child: Text('Select a city and zipcode to view price trends',
                      style: TextStyle(color: Colors.grey)),
                ),
              )
            else
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: _ChartCard(
                      title:
                          '${provider.selectedCity ?? ''} – City Average',
                      points: provider.cityHistory,
                      lineColor: const Color(0xFF1565C0),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: _ChartCard(
                      title:
                          'Zipcode ${provider.selectedZipcode ?? ''} – Trends',
                      points: provider.zipcodeHistory,
                      lineColor: Colors.red,
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

class _ChartCard extends StatelessWidget {
  final String title;
  final List<PricePoint> points;
  final Color lineColor;

  const _ChartCard({
    required this.title,
    required this.points,
    required this.lineColor,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 13,
            color: Color(0xFF1A237E),
          ),
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 260,
          child: points.isEmpty
              ? const Center(
                  child: Text('No data', style: TextStyle(color: Colors.grey)),
                )
              : _PriceLineChart(points: points, lineColor: lineColor),
        ),
      ],
    );
  }
}

class _PriceLineChart extends StatefulWidget {
  final List<PricePoint> points;
  final Color lineColor;

  const _PriceLineChart({required this.points, required this.lineColor});

  @override
  State<_PriceLineChart> createState() => _PriceLineChartState();
}

class _PriceLineChartState extends State<_PriceLineChart> {
  int? _touchedIndex;

  @override
  Widget build(BuildContext context) {
    final spots = _buildSpots(widget.points);
    if (spots.isEmpty) return const SizedBox.shrink();

    final minY = spots.map((s) => s.y).reduce((a, b) => a < b ? a : b);
    final maxY = spots.map((s) => s.y).reduce((a, b) => a > b ? a : b);
    final padding = (maxY - minY) * 0.1;

    // X label interval: show ~6 labels
    final xInterval = (spots.length / 6).clamp(1.0, double.infinity);

    return LineChart(
      LineChartData(
        minY: minY - padding,
        maxY: maxY + padding,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => FlLine(
            color: Colors.grey.withOpacity(0.15),
            strokeWidth: 1,
          ),
        ),
        borderData: FlBorderData(
          show: true,
          border: Border(
            bottom: BorderSide(color: Colors.grey.withOpacity(0.3)),
            left: BorderSide(color: Colors.grey.withOpacity(0.3)),
          ),
        ),
        titlesData: FlTitlesData(
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              interval: xInterval,
              reservedSize: 28,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= widget.points.length) {
                  return const SizedBox.shrink();
                }
                return SideTitleWidget(
                  axisSide: meta.axisSide,
                  child: Text(
                    DateFormat('yyyy').format(widget.points[index].date),
                    style: const TextStyle(
                      fontSize: 10,
                      color: Colors.grey,
                    ),
                  ),
                );
              },
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 72,
              getTitlesWidget: (value, meta) => SideTitleWidget(
                axisSide: meta.axisSide,
                child: Text(
                  _formatPrice(value),
                  style: const TextStyle(fontSize: 10, color: Colors.grey),
                ),
              ),
            ),
          ),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (spots) => spots.map((spot) {
              final index = spot.x.toInt();
              final date = index >= 0 && index < widget.points.length
                  ? DateFormat('MMM yyyy').format(widget.points[index].date)
                  : '';
              return LineTooltipItem(
                '$date\n${_formatPrice(spot.y)}',
                TextStyle(
                  color: widget.lineColor,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              );
            }).toList(),
          ),
          touchCallback: (event, response) {
            setState(() {
              _touchedIndex =
                  response?.lineBarSpots?.first.spotIndex;
            });
          },
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.3,
            color: widget.lineColor,
            barWidth: 2,
            dotData: FlDotData(
              show: _touchedIndex != null,
              checkToShowDot: (spot, barData) =>
                  spot.x.toInt() == _touchedIndex,
            ),
            belowBarData: BarAreaData(
              show: true,
              color: widget.lineColor.withOpacity(0.08),
            ),
          ),
        ],
      ),
    );
  }

  List<FlSpot> _buildSpots(List<PricePoint> points) => points
      .asMap()
      .entries
      .map((e) => FlSpot(e.key.toDouble(), e.value.value))
      .toList();

  String _formatPrice(double v) {
    if (v >= 1000000) return '\$${(v / 1000000).toStringAsFixed(1)}M';
    if (v >= 1000) return '\$${(v / 1000).toStringAsFixed(0)}K';
    return '\$${v.toStringAsFixed(0)}';
  }
}
