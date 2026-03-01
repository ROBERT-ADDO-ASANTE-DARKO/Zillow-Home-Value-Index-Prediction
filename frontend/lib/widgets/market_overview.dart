import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';

class MarketOverviewSection extends StatelessWidget {
  const MarketOverviewSection({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return _SectionCard(
      title: 'Market Overview',
      icon: Icons.bar_chart,
      child: provider.dataState == LoadingState.loading
          ? const _LoadingWidget()
          : provider.metrics == null
              ? const _EmptyWidget()
              : Column(
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: _MetricCard(
                            label: 'Market Volatility',
                            value: NumberFormat.currency(
                              symbol: '\$',
                              decimalDigits: 0,
                            ).format(provider.metrics!.volatility),
                            icon: Icons.show_chart,
                            color: Colors.orange,
                            subtitle: provider.metrics!.volatility > 100000
                                ? 'High'
                                : 'Low',
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: _MetricCard(
                            label: 'Return on Investment',
                            value: NumberFormat.percentPattern()
                                .format(provider.metrics!.roi),
                            icon: Icons.trending_up,
                            color: provider.metrics!.roi >= 0
                                ? Colors.green
                                : Colors.red,
                            subtitle: provider.metrics!.roi > 0.1
                                ? 'Above average'
                                : 'Below average',
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: _MetricCard(
                            label: 'Risk Score',
                            value: provider.metrics!.riskScore
                                .toStringAsFixed(0),
                            icon: Icons.security,
                            color: provider.metrics!.riskScore > 50000
                                ? Colors.red
                                : provider.metrics!.riskScore > 30000
                                    ? Colors.orange
                                    : Colors.green,
                            subtitle: provider.metrics!.riskScore > 50000
                                ? 'High risk'
                                : provider.metrics!.riskScore > 30000
                                    ? 'Moderate'
                                    : 'Low risk',
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;
  final String subtitle;

  const _MetricCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  label,
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 13,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(
              color: color.withOpacity(0.7),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

// ── Shared section container ────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Widget child;

  const _SectionCard({
    required this.title,
    required this.icon,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: const Color(0xFF1565C0), size: 22),
                const SizedBox(width: 10),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }
}

class _LoadingWidget extends StatelessWidget {
  const _LoadingWidget();

  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: CircularProgressIndicator(),
        ),
      );
}

class _EmptyWidget extends StatelessWidget {
  const _EmptyWidget();

  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: Text('No data available',
              style: TextStyle(color: Colors.grey)),
        ),
      );
}
