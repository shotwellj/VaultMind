/**
 * Settings Screen - Server connection, sync status, preferences
 */

import React, { useState, useEffect } from "react";
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, Alert,
} from "react-native";
import { configureServer, getServerUrl, checkConnection } from "../services/api";
import { runSyncCycle, getQueueSize } from "../services/sync";

export default function SettingsScreen() {
  const [serverUrl, setServerUrl] = useState("");
  const [connected, setConnected] = useState(null);
  const [syncing, setSyncing] = useState(false);
  const [queueSize, setQueueSize] = useState(0);

  useEffect(() => {
    (async () => {
      const url = await getServerUrl();
      if (url) setServerUrl(url);
      const q = await getQueueSize();
      setQueueSize(q);
    })();
  }, []);

  const testConnection = async () => {
    if (!serverUrl.trim()) return;
    await configureServer(serverUrl.trim());
    const result = await checkConnection();
    setConnected(result.connected);
    if (result.connected) {
      Alert.alert("Connected", "Successfully connected to your VaultMind server.");
    } else {
      Alert.alert("Connection Failed", result.reason || "Could not reach server.");
    }
  };

  const handleSync = async () => {
    setSyncing(true);
    try {
      const result = await runSyncCycle();
      const pushed = result.push?.pushed || 0;
      const pulled = result.pull?.pulled || 0;
      Alert.alert("Sync Complete", `Pushed ${pushed} items, pulled ${pulled} items.`);
      setQueueSize(await getQueueSize());
    } catch (e) {
      Alert.alert("Sync Failed", e.message);
    } finally {
      setSyncing(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>Server Connection</Text>
      <Text style={styles.hint}>
        Enter your VaultMind server URL. Use your local IP or WireGuard VPN address.
      </Text>
      <TextInput
        style={styles.input}
        value={serverUrl}
        onChangeText={setServerUrl}
        placeholder="http://192.168.1.100:7777"
        placeholderTextColor="#4b5563"
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
      />
      <TouchableOpacity style={styles.btn} onPress={testConnection}>
        <Text style={styles.btnText}>Test Connection</Text>
      </TouchableOpacity>

      {connected !== null && (
        <View style={[styles.statusPill, connected ? styles.statusOk : styles.statusFail]}>
          <Text style={styles.statusText}>
            {connected ? "Connected" : "Not Connected"}
          </Text>
        </View>
      )}

      <Text style={[styles.sectionTitle, { marginTop: 32 }]}>Sync</Text>
      <Text style={styles.hint}>
        {queueSize > 0
          ? `${queueSize} items queued for sync.`
          : "All caught up. No items pending."}
      </Text>
      <TouchableOpacity
        style={[styles.btn, syncing && styles.btnDisabled]}
        onPress={handleSync}
        disabled={syncing}
      >
        <Text style={styles.btnText}>{syncing ? "Syncing..." : "Sync Now"}</Text>
      </TouchableOpacity>

      <Text style={[styles.sectionTitle, { marginTop: 32 }]}>Privacy</Text>
      <View style={styles.privacyCard}>
        <Text style={styles.privacyItem}>No Apple iCloud sync</Text>
        <Text style={styles.privacyItem}>No Google Cloud sync</Text>
        <Text style={styles.privacyItem}>Direct phone-to-server connection</Text>
        <Text style={styles.privacyItem}>HMAC-SHA256 signed messages</Text>
        <Text style={styles.privacyItem}>WireGuard VPN recommended</Text>
      </View>

      <Text style={styles.version}>VaultMind Mobile v1.0.0</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e17" },
  content: { padding: 20, paddingBottom: 40 },
  sectionTitle: { color: "#f9fafb", fontSize: 18, fontWeight: "700", marginBottom: 6 },
  hint: { color: "#9ca3af", fontSize: 13, marginBottom: 12, lineHeight: 18 },
  input: {
    backgroundColor: "#1f2937", color: "#f9fafb", borderRadius: 10,
    padding: 14, fontSize: 15, borderWidth: 1, borderColor: "#374151", marginBottom: 12,
  },
  btn: {
    backgroundColor: "#3b82f6", borderRadius: 10, padding: 14, alignItems: "center",
  },
  btnDisabled: { backgroundColor: "#374151" },
  btnText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  statusPill: {
    alignSelf: "flex-start", paddingHorizontal: 14, paddingVertical: 6,
    borderRadius: 20, marginTop: 12,
  },
  statusOk: { backgroundColor: "#065f46" },
  statusFail: { backgroundColor: "#7f1d1d" },
  statusText: { color: "#f9fafb", fontSize: 13, fontWeight: "500" },
  privacyCard: {
    backgroundColor: "#111827", padding: 16, borderRadius: 12,
    borderWidth: 1, borderColor: "#1f2937",
  },
  privacyItem: { color: "#9ca3af", fontSize: 14, paddingVertical: 4 },
  version: { color: "#4b5563", textAlign: "center", marginTop: 32, fontSize: 12 },
});
