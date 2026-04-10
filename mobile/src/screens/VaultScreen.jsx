/**
 * Vault Screen - Browse indexed documents
 */

import React, { useState, useEffect } from "react";
import {
  View, Text, FlatList, TouchableOpacity, StyleSheet, RefreshControl,
} from "react-native";
import { listFiles } from "../services/api";

export default function VaultScreen() {
  const [files, setFiles] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadFiles = async () => {
    try {
      setError(null);
      const result = await listFiles();
      setFiles(result.files || []);
    } catch (e) {
      setError(e.message);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadFiles();
    setRefreshing(false);
  };

  useEffect(() => { loadFiles(); }, []);

  const renderFile = ({ item }) => (
    <TouchableOpacity style={styles.fileCard}>
      <Text style={styles.fileName}>{item.name || item.source || "Unknown"}</Text>
      <Text style={styles.fileMeta}>
        {item.chunks || "?"} chunks
        {item.indexed_at ? ` | ${new Date(item.indexed_at).toLocaleDateString()}` : ""}
      </Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      {error ? (
        <View style={styles.errorCard}>
          <Text style={styles.errorText}>Cannot reach server: {error}</Text>
          <Text style={styles.errorHint}>Check your VPN connection or server status.</Text>
        </View>
      ) : null}

      <FlatList
        data={files}
        renderItem={renderFile}
        keyExtractor={(item, i) => item.name || String(i)}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#3b82f6" />
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>
            No documents indexed yet. Upload files on the desktop app.
          </Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e17" },
  list: { padding: 16 },
  fileCard: {
    backgroundColor: "#111827", padding: 14, borderRadius: 10,
    marginBottom: 8, borderWidth: 1, borderColor: "#1f2937",
  },
  fileName: { color: "#f9fafb", fontSize: 15, fontWeight: "500" },
  fileMeta: { color: "#6b7280", fontSize: 12, marginTop: 4 },
  errorCard: {
    backgroundColor: "#7f1d1d", margin: 16, padding: 14, borderRadius: 10,
  },
  errorText: { color: "#fca5a5", fontSize: 14, fontWeight: "500" },
  errorHint: { color: "#f87171", fontSize: 12, marginTop: 4 },
  emptyText: { color: "#6b7280", textAlign: "center", marginTop: 40, fontSize: 15 },
});
