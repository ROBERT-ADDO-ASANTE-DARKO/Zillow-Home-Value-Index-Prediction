import 'package:auth0_flutter/auth0_flutter.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

import '../config/constants.dart';
import '../models/models.dart';

/// Wraps the auth0_flutter SDK for login / logout / token refresh.
///
/// On web the SDK uses the Universal Login redirect flow (PKCE).
/// On native (Android / iOS) it uses the Native SDK (device SSO).
class AuthService {
  AuthService()
      : _auth0 = Auth0(AppConstants.auth0Domain, AppConstants.auth0ClientId);

  final Auth0 _auth0;

  // ---------------------------------------------------------------------------
  // Login
  // ---------------------------------------------------------------------------

  /// Initiates the Auth0 Universal Login flow.
  ///
  /// Returns an [AuthUser] on success; throws on cancellation or failure.
  Future<AuthUser> login() async {
    Credentials credentials;

    if (kIsWeb) {
      credentials = await _auth0.webAuthentication().login(
            audience: 'https://${AppConstants.auth0Domain}/api/v2/',
          );
    } else {
      credentials = await _auth0.webAuthentication().login(
            audience: 'https://${AppConstants.auth0Domain}/api/v2/',
          );
    }

    return _credentialsToUser(credentials);
  }

  // ---------------------------------------------------------------------------
  // Logout
  // ---------------------------------------------------------------------------

  Future<void> logout() async {
    await _auth0.webAuthentication().logout();
  }

  // ---------------------------------------------------------------------------
  // Token refresh / restore session
  // ---------------------------------------------------------------------------

  /// Attempts to restore a cached session from the credential manager.
  /// Returns null if no valid session exists.
  Future<AuthUser?> tryRestoreSession() async {
    try {
      final credentials = await _auth0.credentialsManager.credentials();
      return _credentialsToUser(credentials);
    } catch (_) {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  AuthUser _credentialsToUser(Credentials credentials) {
    final profile = credentials.user;
    return AuthUser(
      sub: profile.sub,
      email: profile.email ?? '',
      name: profile.name ?? profile.email ?? 'User',
      pictureUrl: profile.pictureUrl?.toString() ?? '',
      accessToken: credentials.accessToken,
    );
  }
}
