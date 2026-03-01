import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';
import '../widgets/forecast_chart.dart';
import '../widgets/insights_widget.dart';
import '../widgets/map_widget.dart';
import '../widgets/market_overview.dart';
import '../widgets/price_chart.dart';
import '../widgets/sidebar_widget.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: Column(
        children: [
          // ── Top app bar ────────────────────────────────────────────────────
          _AppBanner(provider: provider),

          // ── Main content ───────────────────────────────────────────────────
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Left sidebar
                const SidebarWidget(),

                // Main scrollable content
                Expanded(
                  child: _buildMainContent(provider),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent(AppProvider provider) {
    if (provider.citiesState == LoadingState.loading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text(
              'Loading Zillow data...',
              style: TextStyle(color: Colors.grey, fontSize: 16),
            ),
            SizedBox(height: 8),
            Text(
              'This may take a moment on the first run.',
              style: TextStyle(color: Colors.grey, fontSize: 13),
            ),
          ],
        ),
      );
    }

    if (provider.citiesState == LoadingState.error) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            const Text(
              'Failed to connect to the backend.',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.red,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Make sure the FastAPI server is running at http://localhost:8000',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 8),
            SelectableText(
              provider.errorMessage ?? '',
              style: const TextStyle(color: Colors.grey, fontSize: 12),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const MarketOverviewSection(),
          const SizedBox(height: 20),
          const MapSection(),
          const SizedBox(height: 20),
          const PriceAnalysisSection(),
          const SizedBox(height: 20),
          const ForecastSection(),
          const SizedBox(height: 20),
          const InsightsSection(),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}

class _AppBanner extends StatelessWidget {
  final AppProvider provider;

  const _AppBanner({required this.provider});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 56,
      color: const Color(0xFF0D1642),
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          const Icon(Icons.home_work, color: Colors.white, size: 24),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'Zillow Home Value Index Prediction',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
                letterSpacing: 0.3,
              ),
            ),
          ),
          if (provider.selectedCity != null) ...[
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.location_on,
                      color: Colors.amber, size: 14),
                  const SizedBox(width: 4),
                  Text(
                    '${provider.selectedCity}'
                    '${provider.selectedZipcode != null ? '  •  ${provider.selectedZipcode}' : ''}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(width: 16),
          if (provider.forecastState == LoadingState.loading ||
              provider.dataState == LoadingState.loading)
            const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.amber,
              ),
            ),
        ],
      ),
    );
  }
}
