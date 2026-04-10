/**
 * Alerts Screen - Proactive Intelligence notifications
 */

import React, { useState, useEffect } from "react";
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, RefreshControl,
} from "react-native";
import { getPendingAlerts, markAlertRead } from "../services/api";

const PRIORITY_COLORS = {
  urgent: "#ef4444",
  high: "#f59e0b",
  medium: "#3b82f6",
  low: "#6b7280",
};

export default function AlertsScreen() {
  const [alerts, setAlerts] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const loadAlerts = async () => {
    try {
      const result = await getPendingAlerts();
      setAlerts(result.alerts || []);
    } catch {}
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadAlerts();
    setRefreshing(false);
  };

  const handleTap = async (alert) => {
    if (!alert.read) {
      try { await markAlertRead(alert.id); } catch {}
      setAlerts((prev) =>
        prev.map((a) => (a.id === alert.id ? { ...a, read: true } : a))
      );
    }
  };

  useEffect(() => { loadAlerts(); }, []);

  const renderAlert = ({ item }) => (
    <TouchableOpacity
      style={[styles.alertCard, item.read && styles.alertRead]}
      onPress={() => handleTap(item)}
    >
      <View style={[styles.priorityDot, { backgroundColor: PRIORITY_COLORS[item.priority] || "#6b7280" }]} />
      <View style={styles.alertContent}>
        <Text style={styles.alertTitle}>{item.title}</Text>
        <Text style={styles.alertMsg} numberOfLines={2}>{item.message}</Text>
        <Text style={styles.alertTime}>
          {item.timestamp ? new Date(item.timestamp).toLocaleString() : ""}
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={alerts}
        renderItem={renderAlert}
        keyExtractor={(item) => item.id || String(Math.random())}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>No alerts. VaultMind is watching for changes.</Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e17" },
  list: { padding: 16 },
  alertCard: {
    flexDirection: "row", backgroundColor: "#111827", padding: 14,
    borderRadius: 10, marginBottom: 8, borderWidth: 1, borderColor: "#1f2937",
  },
  alertRead: { opacity: 0.6 },
  priorityDot: {
    width: 10, height: 10, borderRadius: 5, marginTop: 5, marginRight: 12,
  },
  alertContent: { flex: 1 },
  alertTitle: { color: "#f9fafb", fontSize: 15, fontWeight: "600" },
  alertMsg: { color: "#9ca3af", fontSize: 13, marginTop: 3, lineHeight: 18 },
  alertTime: { color: "#4b5563", fontSize: 11, marginTop: 6 },
  emptyText: { color: "#6b7280", textAlign: "center", marginTop: 40, fontSize: 15 },
});
