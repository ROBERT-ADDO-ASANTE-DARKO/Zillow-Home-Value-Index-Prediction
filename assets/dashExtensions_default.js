window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature) {
            return {
                fillColor: feature.properties.fill || '#30363d',
                fillOpacity: 0.72,
                color: '#d4a017',
                weight: 1.5,
                opacity: 0.9,
                dashArray: feature.properties.type === 'prime' ? '4 3' : null,
            };
        },
        function1: function(feature) {
            return {
                fillColor: feature.properties.fill || '#30363d',
                fillOpacity: 0.72,
                color: feature.properties.growth_pct >= 0 ? '#3fb950' : '#f85149',
                weight: 1.5,
                opacity: 0.9,
                dashArray: feature.properties.type === 'prime' ? '4 3' : null,
            };
        },
        function2: function(feature, layer) {
            var p = feature.properties;
            layer.bindTooltip(
                '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:180px;">' +
                '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
                '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + '</div>' +
                '<hr style="border-color:#30363d;margin:5px 0"/>' +
                '<div style="color:#e6edf3;font-size:12px;">AHPI Dec 2024: <b>' + p.ahpi.toFixed(1) + '</b></div>' +
                '<div style="color:#e6edf3;font-size:12px;">GHS/sqm: <b>' + p.ghs_sqm.toLocaleString() + '</b></div>' +
                '<div style="color:#e6edf3;font-size:12px;">USD/sqm: <b>' + p.usd_sqm.toLocaleString() + '</b></div>' +
                '<div style="color:#3fb950;font-size:12px;">USD gain 2010–24: <b>+' + p.usd_pct.toFixed(1) + '%</b></div>' +
                '</div>', {
                    sticky: true,
                    opacity: 1
                }
            );
            layer.on('mouseover', function(e) {
                layer.setStyle({
                    fillOpacity: 0.92,
                    weight: 3
                });
            });
            layer.on('mouseout', function(e) {
                layer.setStyle({
                    fillOpacity: 0.72,
                    weight: 1.5
                });
            });
        },
        function3: function(feature, layer) {
            var p = feature.properties;
            var growthColor = p.growth_pct >= 0 ? '#3fb950' : '#f85149';
            var sign = p.growth_pct >= 0 ? '+' : '';
            layer.bindTooltip(
                '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:190px;">' +
                '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
                '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + '</div>' +
                '<hr style="border-color:#30363d;margin:5px 0"/>' +
                '<div style="color:#e6edf3;font-size:12px;">Forecast AHPI: <b>' + (p.fc_ahpi ? p.fc_ahpi.toFixed(1) : "—") + '</b></div>' +
                '<div style="font-size:13px;font-weight:700;color:' + growthColor + ';">Growth vs 2024: ' + sign + p.growth_pct.toFixed(1) + '%</div>' +
                '</div>', {
                    sticky: true,
                    opacity: 1
                }
            );
            layer.on('mouseover', function(e) {
                layer.setStyle({
                    fillOpacity: 0.92,
                    weight: 3
                });
            });
            layer.on('mouseout', function(e) {
                layer.setStyle({
                    fillOpacity: 0.72,
                    weight: 1.5
                });
            });
        },
        function4: function(feature, layer) {
            var p = feature.properties;
            var proj = p.projected;
            var priceRows = proj ?
                '<div style="color:#58a6ff;font-size:11px;font-style:italic;margin-top:3px;">Projected — AHPI only</div>' :
                '<div style="color:#e6edf3;font-size:12px;">GHS/sqm: <b>' + (p.ghs_sqm ? p.ghs_sqm.toLocaleString() : '—') + '</b></div>' +
                '<div style="color:#e6edf3;font-size:12px;">USD/sqm: <b>' + (p.usd_sqm ? p.usd_sqm.toLocaleString() : '—') + '</b></div>';
            layer.bindTooltip(
                '<div style="background:#161b22;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-family:monospace;min-width:180px;">' +
                '<div style="color:#d4a017;font-weight:700;font-size:13px;margin-bottom:4px;">' + p.name + '</div>' +
                '<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">' + p.type + ' · ' + p.year + '</div>' +
                '<hr style="border-color:#30363d;margin:5px 0"/>' +
                '<div style="color:#e6edf3;font-size:12px;">AHPI' + (proj ? ' (proj)' : '') + ': <b>' + p.ahpi.toFixed(1) + '</b></div>' +
                priceRows +
                '</div>', {
                    sticky: true,
                    opacity: 1
                }
            );
            layer.on('mouseover', function(e) {
                layer.setStyle({
                    fillOpacity: 0.92,
                    weight: 3
                });
            });
            layer.on('mouseout', function(e) {
                layer.setStyle({
                    fillOpacity: 0.72,
                    weight: 1.5
                });
            });
        }
    }
});