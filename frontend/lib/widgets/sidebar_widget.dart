import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_provider.dart';

class SidebarWidget extends StatefulWidget {
  const SidebarWidget({super.key});

  @override
  State<SidebarWidget> createState() => _SidebarWidgetState();
}

class _SidebarWidgetState extends State<SidebarWidget> {
  final _eventNameController = TextEditingController();
  DateTime? _eventDate;
  double _eventImpact = 0;
  bool _showEventControls = false;

  @override
  void dispose() {
    _eventNameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();

    return Container(
      width: 300,
      color: const Color(0xFF1A237E),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header ──────────────────────────────────────────────────────────
          Container(
            padding: const EdgeInsets.all(20),
            color: const Color(0xFF0D1642),
            child: const Row(
              children: [
                Icon(Icons.home, color: Colors.white, size: 28),
                SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'ZHVI Prediction',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),

          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionLabel('Analysis Parameters'),
                  const SizedBox(height: 12),

                  // ── City selector ─────────────────────────────────────────
                  _label('Select City'),
                  const SizedBox(height: 6),
                  _buildDropdown<String>(
                    value: provider.selectedCity,
                    items: provider.cities,
                    onChanged: provider.citiesState == LoadingState.loading
                        ? null
                        : (v) {
                            if (v != null) {
                              context.read<AppProvider>().selectCity(v);
                            }
                          },
                    hint: 'Loading cities...',
                  ),

                  const SizedBox(height: 16),

                  // ── Zipcode selector ──────────────────────────────────────
                  _label('Select Zipcode'),
                  const SizedBox(height: 6),
                  _buildDropdown<String>(
                    value: provider.selectedZipcode,
                    items: provider.zipcodes,
                    onChanged: provider.dataState == LoadingState.loading
                        ? null
                        : (v) {
                            if (v != null) {
                              context.read<AppProvider>().selectZipcode(v);
                            }
                          },
                    hint: 'Select a city first',
                  ),

                  const SizedBox(height: 20),

                  // ── Prediction years slider ───────────────────────────────
                  _label('Prediction Timeframe'),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '${provider.years} ${provider.years == 1 ? 'year' : 'years'}',
                        style: const TextStyle(
                          color: Colors.amber,
                          fontWeight: FontWeight.bold,
                          fontSize: 15,
                        ),
                      ),
                    ],
                  ),
                  Slider(
                    value: provider.years.toDouble(),
                    min: 1,
                    max: 20,
                    divisions: 19,
                    activeColor: Colors.amber,
                    inactiveColor: Colors.white24,
                    label: '${provider.years} yr',
                    onChanged: (v) {
                      context.read<AppProvider>().setYears(v.toInt());
                    },
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _smallLabel('1 yr'),
                      _smallLabel('20 yr'),
                    ],
                  ),

                  const SizedBox(height: 20),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),

                  // ── Event analysis ────────────────────────────────────────
                  _sectionLabel('Event Analysis'),
                  const SizedBox(height: 10),
                  _label('Hypothetical Event Name'),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _eventNameController,
                    style: const TextStyle(color: Colors.white),
                    decoration: InputDecoration(
                      hintText: 'e.g., New infrastructure project',
                      hintStyle: const TextStyle(color: Colors.white38),
                      filled: true,
                      fillColor: Colors.white12,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 10,
                      ),
                    ),
                    onChanged: (v) {
                      setState(() => _showEventControls = v.isNotEmpty);
                    },
                  ),

                  if (_showEventControls) ...[
                    const SizedBox(height: 12),
                    _label('Event Date'),
                    const SizedBox(height: 6),
                    InkWell(
                      onTap: () async {
                        final picked = await showDatePicker(
                          context: context,
                          initialDate: DateTime.now(),
                          firstDate: DateTime(2000),
                          lastDate: DateTime(2030),
                        );
                        if (picked != null) {
                          setState(() => _eventDate = picked);
                        }
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 10,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white12,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.calendar_today,
                                color: Colors.white54, size: 16),
                            const SizedBox(width: 8),
                            Text(
                              _eventDate != null
                                  ? '${_eventDate!.year}-'
                                      '${_eventDate!.month.toString().padLeft(2, '0')}-'
                                      '${_eventDate!.day.toString().padLeft(2, '0')}'
                                  : 'Pick a date',
                              style: TextStyle(
                                color: _eventDate != null
                                    ? Colors.white
                                    : Colors.white38,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    _label('Event Impact: ${_eventImpact.toInt()}'),
                    Slider(
                      value: _eventImpact,
                      min: -100,
                      max: 100,
                      divisions: 200,
                      activeColor: _eventImpact >= 0
                          ? Colors.greenAccent
                          : Colors.redAccent,
                      inactiveColor: Colors.white24,
                      label: _eventImpact.toInt().toString(),
                      onChanged: (v) => setState(() => _eventImpact = v),
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _smallLabel('-100'),
                        _smallLabel('+100'),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton(
                            onPressed: _eventDate != null
                                ? () {
                                    context.read<AppProvider>().applyEvent(
                                          _eventNameController.text,
                                          _eventDate!,
                                          _eventImpact.toInt(),
                                        );
                                  }
                                : null,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.amber,
                              foregroundColor: Colors.black87,
                            ),
                            child: const Text('Apply Event'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: const Icon(Icons.clear, color: Colors.white70),
                          tooltip: 'Clear event',
                          onPressed: () {
                            _eventNameController.clear();
                            setState(() {
                              _showEventControls = false;
                              _eventDate = null;
                              _eventImpact = 0;
                            });
                            context.read<AppProvider>().clearEvent();
                          },
                        ),
                      ],
                    ),
                  ],

                  const SizedBox(height: 20),
                  const Divider(color: Colors.white24),

                  // ── Glossary ──────────────────────────────────────────────
                  Theme(
                    data: Theme.of(context).copyWith(
                      dividerColor: Colors.transparent,
                    ),
                    child: ExpansionTile(
                      title: const Row(
                        children: [
                          Icon(Icons.menu_book, color: Colors.white70, size: 18),
                          SizedBox(width: 8),
                          Text(
                            'Glossary',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                      iconColor: Colors.white70,
                      collapsedIconColor: Colors.white54,
                      children: [
                        _glossaryEntry('Market Volatility',
                            'How much home prices fluctuate over time. Higher = greater price swings.'),
                        _glossaryEntry('Return on Investment (ROI)',
                            'Percentage increase in home prices over the full historical period.'),
                        _glossaryEntry('Risk Score',
                            'Composite score: 60% volatility + 40% ROI. Lower is safer.'),
                        _glossaryEntry('Forecast',
                            'Prophet model prediction of future home prices based on historical trends.'),
                        _glossaryEntry('Event Impact',
                            'Simulated effect of a hypothetical event on prices (+/- 100 scale).'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text) => Text(
        text.toUpperCase(),
        style: const TextStyle(
          color: Colors.white54,
          fontSize: 11,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.2,
        ),
      );

  Widget _label(String text) => Text(
        text,
        style: const TextStyle(color: Colors.white70, fontSize: 13),
      );

  Widget _smallLabel(String text) => Text(
        text,
        style: const TextStyle(color: Colors.white38, fontSize: 11),
      );

  Widget _buildDropdown<T>({
    required T? value,
    required List<T> items,
    required ValueChanged<T?>? onChanged,
    String hint = 'Select...',
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.white12,
        borderRadius: BorderRadius.circular(8),
      ),
      child: DropdownButton<T>(
        value: value,
        isExpanded: true,
        dropdownColor: const Color(0xFF1A237E),
        style: const TextStyle(color: Colors.white),
        underline: const SizedBox.shrink(),
        icon: const Icon(Icons.keyboard_arrow_down, color: Colors.white54),
        hint: Text(hint, style: const TextStyle(color: Colors.white38)),
        items: items.map((item) {
          return DropdownMenuItem<T>(
            value: item,
            child: Text(
              item.toString(),
              style: const TextStyle(color: Colors.white),
              overflow: TextOverflow.ellipsis,
            ),
          );
        }).toList(),
        onChanged: onChanged,
      ),
    );
  }

  Widget _glossaryEntry(String term, String definition) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              term,
              style: const TextStyle(
                color: Colors.amber,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              definition,
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          ],
        ),
      );
}
