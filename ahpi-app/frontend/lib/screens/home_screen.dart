import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';
import 'districts_screen.dart';
import 'forecast_screen.dart';
import 'map_screen.dart';
import 'overview_screen.dart';
import 'prime_screen.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  int _selectedIndex = 0;

  static const _tabs = [
    (icon: Icons.home_outlined, label: 'Overview'),
    (icon: Icons.location_city_outlined, label: 'Districts'),
    (icon: Icons.star_outline, label: 'Prime'),
    (icon: Icons.auto_graph, label: 'Forecasts'),
    (icon: Icons.map_outlined, label: 'Map'),
  ];

  static const _screens = [
    OverviewScreen(),
    DistrictsScreen(),
    PrimeScreen(),
    ForecastScreen(),
    MapScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final user = authState.user!;
    final isWide = MediaQuery.of(context).size.width >= 768;

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        elevation: 0,
        titleSpacing: 20,
        title: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: const Color(0xFF1C6B2A),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.home_work_outlined,
                  color: Colors.white, size: 18),
            ),
            const SizedBox(width: 10),
            const Text(
              'Accra Home Price Index',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600),
            ),
          ],
        ),
        actions: [
          // User avatar + name
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (user.pictureUrl.isNotEmpty)
                  CircleAvatar(
                    backgroundImage: NetworkImage(user.pictureUrl),
                    radius: 14,
                  )
                else
                  const CircleAvatar(
                    backgroundColor: Color(0xFF1C6B2A),
                    radius: 14,
                    child: Icon(Icons.person, color: Colors.white, size: 16),
                  ),
                const SizedBox(width: 6),
                if (isWide)
                  Text(user.name,
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 13)),
              ],
            ),
          ),
          // Logout
          IconButton(
            tooltip: 'Sign out',
            icon: const Icon(Icons.logout, color: Colors.grey, size: 20),
            onPressed: () => ref.read(authProvider.notifier).logout(),
          ),
          const SizedBox(width: 4),
        ],
        bottom: isWide
            ? null
            : PreferredSize(
                preferredSize: const Size.fromHeight(48),
                child: _TabBar(
                  selectedIndex: _selectedIndex,
                  onTabSelected: (i) => setState(() => _selectedIndex = i),
                  tabs: _tabs,
                ),
              ),
      ),
      body: isWide
          ? Row(
              children: [
                // Side navigation rail
                NavigationRail(
                  backgroundColor: const Color(0xFF161B22),
                  selectedIndex: _selectedIndex,
                  onDestinationSelected: (i) =>
                      setState(() => _selectedIndex = i),
                  labelType: NavigationRailLabelType.all,
                  selectedIconTheme:
                      const IconThemeData(color: Color(0xFF4CAF50)),
                  selectedLabelTextStyle:
                      const TextStyle(color: Color(0xFF4CAF50), fontSize: 11),
                  unselectedIconTheme:
                      const IconThemeData(color: Colors.grey),
                  unselectedLabelTextStyle:
                      const TextStyle(color: Colors.grey, fontSize: 11),
                  destinations: _tabs
                      .map((t) => NavigationRailDestination(
                            icon: Icon(t.icon),
                            label: Text(t.label),
                          ))
                      .toList(),
                ),
                const VerticalDivider(
                    width: 1, color: Color(0xFF30363D)),
                // Main content
                Expanded(child: _screens[_selectedIndex]),
              ],
            )
          : _screens[_selectedIndex],
    );
  }
}

// ---------------------------------------------------------------------------
// Mobile tab bar
// ---------------------------------------------------------------------------

class _TabBar extends StatelessWidget {
  const _TabBar({
    required this.selectedIndex,
    required this.onTabSelected,
    required this.tabs,
  });

  final int selectedIndex;
  final ValueChanged<int> onTabSelected;
  final List<({IconData icon, String label})> tabs;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 48,
      child: Row(
        children: tabs.asMap().entries.map((e) {
          final isSelected = e.key == selectedIndex;
          return Expanded(
            child: GestureDetector(
              onTap: () => onTabSelected(e.key),
              child: Container(
                color: Colors.transparent,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      e.value.icon,
                      size: 18,
                      color: isSelected
                          ? const Color(0xFF4CAF50)
                          : Colors.grey,
                    ),
                    Text(
                      e.value.label,
                      style: TextStyle(
                        fontSize: 10,
                        color: isSelected
                            ? const Color(0xFF4CAF50)
                            : Colors.grey,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}
