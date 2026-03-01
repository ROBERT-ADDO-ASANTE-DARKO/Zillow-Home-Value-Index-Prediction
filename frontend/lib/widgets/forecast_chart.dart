import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../models/app_models.dart';
import '../providers/app_provider.dart';

class ForecastSection extends StatelessWidget {
  const ForecastSection({super.key});

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
            // ── Header ───────────────────────────────────────────────────────
            Row(
              children: [
                const Icon(Icons.auto_graph, color: Color(0xFF1565C0), size: 22),
                const SizedBox(width: 10),
                Text(
                  'Price Forecast – ${provider.years} ${provider.years == 1 ? 'Year' : 'Years'}',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
                const Spacer(),
                if (provider.forecastState == LoadingState.loading)
                  const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
              ],
            ),

            if (provider.eventName != null &&
                provider.eventName!.isNotEmpty) ...[
              const SizedBox(height: 8),
              Chip(
                avatar: const Icon(Icons.event, size: 14),
                label: Text(
                  'Event: ${provider.eventName}'
                  ' (impact: ${provider.eventImpact > 0 ? '+' : ''}${provider.eventImpact})',
                  style: const TextStyle(fontSize: 12),
                ),
                backgroundColor: Colors.amber.withOpacity(0.15),
              ),
            ],

            const SizedBox(height: 16),

            // ── Legend ───────────────────────────────────────────────────────
            _Legend(),

            const SizedBox(height: 12),

            // ── Chart ────────────────────────────────────────────────────────
            if (provider.forecastState == LoadingState.loading &&
                provider.forecast == null)
              const SizedBox(
                height: 320,
                child: Center(child: CircularProgressIndicator()),
              )
            else if (provider.forecast == null)
              const SizedBox(
                height: 200,
                child: Center(
                  child: Text(
                    'Forecast will appear after selecting a zipcode.',
                    style: TextStyle(color: Colors.grey),
                  ),
                ),
              )
            else
              SizedBox(
                height: 320,
                child: _ForecastLineChart(
                  historical: provider.zipcodeHistory,
                  forecast: provider.forecast!,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 20,
      runSpacing: 8,
      children: [
        _legendItem(Colors.red, 'Historical Actual', solid: true),
        _legendItem(const Color(0xFF1565C0), 'Prophet Forecast', solid: false),
        _legendItem(
            const Color(0xFF1565C0).withOpacity(0.25), 'Confidence Interval',
            solid: true),
      ],
    );
  }

  Widget _legendItem(Color color, String label, {required bool solid}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 24,
          height: 3,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(2),
          ),
          child: solid
              ? null
              : CustomPaint(painter: _DashPainter(color: color)),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.grey)),
      ],
    );
  }
}

class _DashPainter extends CustomPainter {
  final Color color;
  _DashPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2;
    double x = 0;
    while (x < size.width) {
      canvas.drawLine(Offset(x, size.height / 2),
          Offset((x + 4).clamp(0, size.width), size.height / 2), paint);
      x += 8;
    }
  }

  @override
  bool shouldRepaint(_) => false;
}

class _ForecastLineChart extends StatelessWidget {
  final List<PricePoint> historical;
  final ForecastResult forecast;

  const _ForecastLineChart({
    required this.historical,
    required this.forecast,
  });

  @override
  Widget build(BuildContext context) {
    // We use a unified x-axis based on the forecast date list (which covers
    // the full historical + future range sampled by the API).
    final allDates = forecast.dates;
    if (allDates.isEmpty) return const SizedBox.shrink();

    // Build forecast spots indexed by position
    final lowerSpots = _indexedSpots(forecast.lower);
    final upperSpots = _indexedSpots(forecast.upper);
    final forecastSpots = _indexedSpots(forecast.forecast);

    // Map historical actual prices onto the forecast date axis
    final historicalSpots = _mapHistoricalToForecastAxis(historical, allDates);

    // Determine Y-axis range from all data
    final allValues = [
      ...forecast.lower,
      ...forecast.upper,
      ...historical.map((p) => p.value),
    ];
    final minY = allValues.reduce((a, b) => a < b ? a : b);
    final maxY = allValues.reduce((a, b) => a > b ? a : b);
    final yPad = (maxY - minY) * 0.08;

    // X label interval: ~7 labels across the date range
    final xInterval = (allDates.length / 7).clamp(1.0, double.infinity);

    return LineChart(
      LineChartData(
        minY: minY - yPad,
        maxY: maxY + yPad,
        clipData: const FlClipData.all(),
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
                if (index < 0 || index >= allDates.length) {
                  return const SizedBox.shrink();
                }
                return SideTitleWidget(
                  axisSide: meta.axisSide,
                  child: Text(
                    DateFormat('yyyy').format(allDates[index]),
                    style: const TextStyle(fontSize: 10, color: Colors.grey),
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
                  _fmtPrice(value),
                  style: const TextStyle(fontSize: 10, color: Colors.grey),
                ),
              ),
            ),
          ),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                final index = spot.x.toInt();
                final date = index >= 0 && index < allDates.length
                    ? DateFormat('MMM yyyy').format(allDates[index])
                    : '';
                String seriesLabel;
                Color color;
                switch (spot.barIndex) {
                  case 2:
                    seriesLabel = 'Forecast';
                    color = const Color(0xFF1565C0);
                    break;
                  case 3:
                    seriesLabel = 'Actual';
                    color = Colors.red;
                    break;
                  default:
                    return null;
                }
                return LineTooltipItem(
                  '$seriesLabel\n$date\n${_fmtPrice(spot.y)}',
                  TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                );
              }).whereType<LineTooltipItem>().toList();
            },
          ),
        ),
        betweenBarsData: [
          // Fill confidence interval between lower (0) and upper (1) bands
          BetweenBarsData(
            fromIndex: 0,
            toIndex: 1,
            color: const Color(0xFF1565C0).withOpacity(0.15),
          ),
        ],
        lineBarsData: [
          // 0 – lower bound (invisible line, used for BetweenBarsData)
          LineChartBarData(
            spots: lowerSpots,
            color: Colors.transparent,
            barWidth: 0,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(show: false),
          ),
          // 1 – upper bound (invisible line, used for BetweenBarsData)
          LineChartBarData(
            spots: upperSpots,
            color: Colors.transparent,
            barWidth: 0,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(show: false),
          ),
          // 2 – Prophet forecast line (dashed blue)
          LineChartBarData(
            spots: forecastSpots,
            isCurved: true,
            curveSmoothness: 0.25,
            color: const Color(0xFF1565C0),
            barWidth: 2,
            dashArray: [6, 4],
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(show: false),
          ),
          // 3 – Historical actual (solid red)
          LineChartBarData(
            spots: historicalSpots,
            isCurved: true,
            curveSmoothness: 0.3,
            color: Colors.red,
            barWidth: 2,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(show: false),
          ),
        ],
      ),
    );
  }

  List<FlSpot> _indexedSpots(List<double> values) =>
      values.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value)).toList();

  /// Map each historical PricePoint to its nearest index in [forecastDates].
  List<FlSpot> _mapHistoricalToForecastAxis(
    List<PricePoint> historical,
    List<DateTime> forecastDates,
  ) {
    if (historical.isEmpty || forecastDates.isEmpty) return [];

    final spots = <FlSpot>[];
    for (final point in historical) {
      // Find the forecast date index closest in time to this historical point
      int bestIndex = 0;
      int bestDiff = (forecastDates.first.difference(point.date)).inDays.abs();
      for (int i = 1; i < forecastDates.length; i++) {
        final diff = forecastDates[i].difference(point.date).inDays.abs();
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIndex = i;
        }
        if (diff > bestDiff) break; // sorted in time, can early-exit
      }
      spots.add(FlSpot(bestIndex.toDouble(), point.value));
    }
    return spots;
  }

  String _fmtPrice(double v) {
    if (v >= 1000000) return '\$${(v / 1000000).toStringAsFixed(1)}M';
    if (v >= 1000) return '\$${(v / 1000).toStringAsFixed(0)}K';
    return '\$${v.toStringAsFixed(0)}';
  }
}
