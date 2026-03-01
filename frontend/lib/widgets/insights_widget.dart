import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';

class InsightsSection extends StatelessWidget {
  const InsightsSection({super.key});

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
                Icon(Icons.insights, color: Color(0xFF1565C0), size: 22),
                SizedBox(width: 10),
                Text(
                  'Key Insights',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (provider.forecastState == LoadingState.loading ||
                provider.dataState == LoadingState.loading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (provider.forecast == null || provider.metrics == null)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('No insights available yet',
                      style: TextStyle(color: Colors.grey)),
                ),
              )
            else
              _buildInsights(provider),
          ],
        ),
      ),
    );
  }

  Widget _buildInsights(AppProvider provider) {
    final forecast = provider.forecast!;
    final metrics = provider.metrics!;
    final city = provider.selectedCity ?? '';
    final zipcode = provider.selectedZipcode ?? '';

    double growthRate = 0;
    if (forecast.forecast.isNotEmpty) {
      final first = forecast.forecast.first;
      final last = forecast.forecast.last;
      if (first != 0) growthRate = (last - first) / first;
    }

    final fmt = NumberFormat.currency(symbol: '\$', decimalDigits: 0);
    final pctFmt = NumberFormat.percentPattern();

    final volatilityHigh = metrics.volatility > 100000;
    final roiAboveAvg = metrics.roi > 0.1;
    final growthFavorable = growthRate > 0.1;
    final riskHigh = metrics.riskScore > 50000;
    final riskModerate = metrics.riskScore > 30000;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: _InsightCard(
            title: 'Market Trends',
            icon: Icons.trending_up,
            color: Colors.blue,
            items: [
              _InsightItem(
                label: 'City',
                value: city,
                icon: Icons.location_city,
              ),
              _InsightItem(
                label: 'Zipcode',
                value: zipcode,
                icon: Icons.pin_drop,
              ),
              _InsightItem(
                label: 'Expected Growth (${provider.years}yr)',
                value: pctFmt.format(growthRate),
                icon: Icons.show_chart,
                valueColor:
                    growthRate >= 0 ? Colors.green : Colors.red,
              ),
              _InsightItem(
                label: 'Market Volatility',
                value: volatilityHigh ? 'High' : 'Low',
                icon: Icons.waves,
                valueColor: volatilityHigh ? Colors.orange : Colors.green,
              ),
              _InsightItem(
                label: 'ROI Performance',
                value: roiAboveAvg ? 'Above average' : 'Below average',
                icon: Icons.percent,
                valueColor: roiAboveAvg ? Colors.green : Colors.orange,
              ),
              if (forecast.forecast.isNotEmpty)
                _InsightItem(
                  label: 'Forecast (${provider.years}yr)',
                  value: fmt.format(forecast.forecast.last),
                  icon: Icons.home_work,
                  valueColor: const Color(0xFF1565C0),
                ),
            ],
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: _InsightCard(
            title: 'Risk Assessment',
            icon: Icons.shield,
            color: riskHigh
                ? Colors.red
                : riskModerate
                    ? Colors.orange
                    : Colors.green,
            items: [
              _InsightItem(
                label: 'Risk Level',
                value: riskHigh
                    ? 'High'
                    : riskModerate
                        ? 'Moderate'
                        : 'Low',
                icon: Icons.warning_amber,
                valueColor: riskHigh
                    ? Colors.red
                    : riskModerate
                        ? Colors.orange
                        : Colors.green,
              ),
              _InsightItem(
                label: 'Market Stability',
                value: volatilityHigh ? 'Volatile' : 'Stable',
                icon: Icons.balance,
                valueColor: volatilityHigh ? Colors.orange : Colors.green,
              ),
              _InsightItem(
                label: 'Investment Outlook',
                value: growthFavorable
                    ? 'Favorable'
                    : growthRate > 0
                        ? 'Moderate'
                        : 'Cautious',
                icon: Icons.attach_money,
                valueColor: growthFavorable
                    ? Colors.green
                    : growthRate > 0
                        ? Colors.orange
                        : Colors.red,
              ),
              _InsightItem(
                label: 'Volatility (σ)',
                value: fmt.format(metrics.volatility),
                icon: Icons.multiline_chart,
              ),
              _InsightItem(
                label: 'Total ROI',
                value: pctFmt.format(metrics.roi),
                icon: Icons.savings,
                valueColor: metrics.roi >= 0 ? Colors.green : Colors.red,
              ),
              _InsightItem(
                label: 'Risk Score',
                value: metrics.riskScore.toStringAsFixed(0),
                icon: Icons.score,
                valueColor: riskHigh
                    ? Colors.red
                    : riskModerate
                        ? Colors.orange
                        : Colors.green,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _InsightCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Color color;
  final List<_InsightItem> items;

  const _InsightCard({
    required this.title,
    required this.icon,
    required this.color,
    required this.items,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                ),
              ),
            ],
          ),
          const Divider(height: 20),
          ...items.map((item) => _buildRow(item)),
        ],
      ),
    );
  }

  Widget _buildRow(_InsightItem item) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Icon(item.icon, size: 15, color: Colors.grey),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              item.label,
              style: const TextStyle(color: Colors.black54, fontSize: 13),
            ),
          ),
          Text(
            item.value,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 13,
              color: item.valueColor ?? Colors.black87,
            ),
          ),
        ],
      ),
    );
  }
}

class _InsightItem {
  final String label;
  final String value;
  final IconData icon;
  final Color? valueColor;

  const _InsightItem({
    required this.label,
    required this.value,
    required this.icon,
    this.valueColor,
  });
}
