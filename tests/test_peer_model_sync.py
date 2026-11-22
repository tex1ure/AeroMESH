"""
Automated unit & integration tests for Automatic Multi-Node Model Check & Peer Model Sync.
Covers:
  - Deterministic GGUF header inspection & fingerprinting (head/mid/tail mutation sensitivity)
  - Strict filename confinement & SSRF host validation
  - Challenge-response HMAC-SHA256 authentication (fail-closed on wrong/empty secret)
  - Pure multi-node comparison matrix classification (verified, missing, mismatch, pull source)
  - End-to-end TCP push, pull, and resumable .part recovery between two live PeerModelSyncServer nodes
"""

import asyncio
import os
import struct
import tempfile
import unittest

from peer_model_sync import (
    SYNC_PART_SUFFIX,
    PeerModelSyncServer,
    PeerSyncJobManager,
    build_cluster_model_matrix,
    build_local_model_manifest,
    fetch_peer_manifest,
    inspect_gguf_file,
    resolve_confined_gguf,
    validate_gguf_filename,
    validate_peer_host,
)


def _write_fake_gguf(path: str, payload_bytes: bytes = b"\x11" * 4096, tensor_count: int = 128, kv_count: int = 24) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, tensor_count, kv_count)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload_bytes)


class TestPeerModelSyncUnit(unittest.TestCase):
    def test_gguf_inspection_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "DeepSeek-R1-14B-Q4_K_M.gguf")
            p2 = os.path.join(tmp, "DeepSeek-R1-14B-Q4_K_M_copy.gguf")
            p3 = os.path.join(tmp, "corrupt.gguf")

            body = os.urandom(2 * 1024 * 1024 + 512 * 1024)  # 2.5 MiB -> exercises head + mid + tail
            _write_fake_gguf(p1, body, tensor_count=291, kv_count=32)
            _write_fake_gguf(p2, body, tensor_count=291, kv_count=32)

            # Mutate a single byte near the middle
            mutated = bytearray(body)
            mutated[len(mutated) // 2] ^= 0xFF
            _write_fake_gguf(p3, bytes(mutated), tensor_count=291, kv_count=32)

            i1 = inspect_gguf_file(p1)
            i2 = inspect_gguf_file(p2)
            i3 = inspect_gguf_file(p3)

            self.assertTrue(i1["gguf_valid"])
            self.assertEqual(i1["gguf_version"], 3)
            self.assertEqual(i1["tensor_count"], 291)
            self.assertEqual(i1["quant"], "Q4_K_M")
            self.assertEqual(i1["fingerprint"], i2["fingerprint"])
            self.assertNotEqual(i1["fingerprint"], i3["fingerprint"])

    def test_non_gguf_flagged_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "fake.gguf")
            with open(bad, "wb") as f:
                f.write(b"NOT_GGUF_DATA_12345678901234567890")
            info = inspect_gguf_file(bad)
            self.assertFalse(info["gguf_valid"])

    def test_security_filename_and_host_validation(self):
        for bad_name in ("../secret.gguf", "a/b.gguf", "a\\b.gguf", "model.bin", ".aeromesh_ephemeral_1.gguf", "x.stage1.gguf", ""):
            with self.assertRaises(ValueError, msg=f"Should reject {bad_name!r}"):
                validate_gguf_filename(bad_name)

        self.assertEqual(validate_gguf_filename("Qwen2.5-7B-Q4_K_M.gguf"), "Qwen2.5-7B-Q4_K_M.gguf")

        with self.assertRaises(ValueError):
            validate_peer_host("169.254.169.254:50054")
        with self.assertRaises(ValueError):
            validate_peer_host("0.0.0.0:50054")
        with self.assertRaises(ValueError):
            validate_peer_host("224.0.0.1:50054")
        with self.assertRaises(ValueError):
            validate_peer_host("192.168.1.5:80")
        self.assertEqual(validate_peer_host("100.64.0.29"), ("100.64.0.29", 50054))

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "models"), exist_ok=True)
            _write_fake_gguf(os.path.join(tmp, "outside.gguf"))
            with self.assertRaises(ValueError):
                resolve_confined_gguf("../outside.gguf", [os.path.join(tmp, "models")])

    def test_cluster_model_matrix_classifications(self):
        local = [
            {"name": "match.gguf", "quant": "Q4_K_M", "size_bytes": 1000, "size_label": "1.0 KB", "fingerprint": "a" * 64, "fingerprint_short": "AAAAAAAAAAAA"},
            {"name": "only-local.gguf", "quant": "Q8_0", "size_bytes": 2000, "size_label": "2.0 KB", "fingerprint": "b" * 64, "fingerprint_short": "BBBBBBBBBBBB"},
            {"name": "conflict.gguf", "quant": "Q4_K_M", "size_bytes": 3000, "size_label": "3.0 KB", "fingerprint": "c" * 64, "fingerprint_short": "CCCCCCCCCCCC"},
        ]
        peers = [
            {
                "id": "p1",
                "node_name": "Worker-RTX4060",
                "host": "100.64.0.29",
                "worker_endpoint": "100.64.0.29:50052",
                "sync_port": 50054,
                "status": "online",
                "models": [
                    {"name": "match.gguf", "quant": "Q4_K_M", "size_bytes": 1000, "size_label": "1.0 KB", "fingerprint": "a" * 64, "fingerprint_short": "AAAAAAAAAAAA"},
                    {"name": "conflict.gguf", "quant": "Q4_K_M", "size_bytes": 3000, "size_label": "3.0 KB", "fingerprint": "d" * 64, "fingerprint_short": "DDDDDDDDDDDD"},
                    {"name": "only-peer.gguf", "quant": "Q4_K_M", "size_bytes": 4000, "size_label": "4.0 KB", "fingerprint": "e" * 64, "fingerprint_short": "EEEEEEEEEEEE"},
                ],
            }
        ]
        report = build_cluster_model_matrix(local, peers, target_model="conflict.gguf")
        by_name = {m["name"]: m for m in report["models"]}
        self.assertEqual(by_name["match.gguf"]["overall_status"], "ready")
        self.assertEqual(by_name["only-local.gguf"]["overall_status"], "needs_sync")
        self.assertEqual(by_name["conflict.gguf"]["overall_status"], "mismatch")
        self.assertEqual(by_name["only-peer.gguf"]["overall_status"], "missing_locally")
        self.assertEqual(report["selected"]["name"], "conflict.gguf")
        self.assertEqual(report["selected"]["overall_status"], "mismatch")


class TestPeerModelSyncEndToEnd(unittest.TestCase):
    def test_live_tcp_auth_manifest_push_and_resumable_pull(self):
        async def _scenario():
            with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
                secret = "0123456789abcdef0123456789abcdef"
                model_push = "PushModel-Q4_K_M.gguf"
                model_pull = "PullModel-Q8_0.gguf"
                payload_push = os.urandom(384 * 1024)
                payload_pull = os.urandom(512 * 1024)

                path_a_push = os.path.join(dir_a, model_push)
                path_b_pull = os.path.join(dir_b, model_pull)
                _write_fake_gguf(path_a_push, payload_push)
                _write_fake_gguf(path_b_pull, payload_pull)

                srv_b = PeerModelSyncServer(
                    get_secret=lambda: secret,
                    get_models_dirs=lambda: [dir_b],
                    get_primary_dir=lambda: dir_b,
                    get_node_meta=lambda: {"node_name": "Node-B", "gpu": "RTX 4060", "cluster_fingerprint": "KEY-TEST-0001"},
                )
                port_b = await srv_b.start(bind_host="127.0.0.1", port=0)

                try:
                    # 1. Wrong secret must fail closed with PermissionError
                    with self.assertRaises(PermissionError):
                        await fetch_peer_manifest("127.0.0.1", port_b, "wrong-secret-value")

                    # 2. Valid secret fetches Node-B manifest
                    manifest = await fetch_peer_manifest("127.0.0.1", port_b, secret)
                    self.assertEqual(manifest["node_name"], "Node-B")
                    self.assertEqual([m["name"] for m in manifest["models"]], [model_pull])

                    jobs = PeerSyncJobManager()

                    # 3. Live PUSH (Node A -> Node B)
                    push_job = jobs.start_push("Node-B", "127.0.0.1", port_b, model_push, path_a_push, secret)
                    for _ in range(80):
                        cur = jobs.get_job(push_job["id"])
                        if cur["status"] in ("completed", "error"):
                            break
                        await asyncio.sleep(0.05)
                    final_push = jobs.get_job(push_job["id"])
                    self.assertEqual(final_push["status"], "completed", final_push.get("error"))
                    self.assertEqual(final_push["percent"], 100.0)

                    # Verify exact fingerprint match on Node B after atomic swap
                    dst_b_push = os.path.join(dir_b, model_push)
                    self.assertTrue(os.path.isfile(dst_b_push))
                    self.assertEqual(
                        inspect_gguf_file(path_a_push)["fingerprint"],
                        inspect_gguf_file(dst_b_push)["fingerprint"],
                    )

                    # 4. Live RESUMABLE PULL (Node B -> Node A) starting from a pre-existing 128 KiB .part file
                    dst_a_pull = os.path.join(dir_a, model_pull)
                    part_a_pull = dst_a_pull + SYNC_PART_SUFFIX
                    with open(path_b_pull, "rb") as src_f, open(part_a_pull, "wb") as part_f:
                        part_f.write(src_f.read(128 * 1024))

                    pull_job = jobs.start_pull("Node-B", "127.0.0.1", port_b, model_pull, dir_a, secret)
                    for _ in range(80):
                        cur = jobs.get_job(pull_job["id"])
                        if cur["status"] in ("completed", "error"):
                            break
                        await asyncio.sleep(0.05)
                    final_pull = jobs.get_job(pull_job["id"])
                    self.assertEqual(final_pull["status"], "completed", final_pull.get("error"))
                    self.assertEqual(final_pull["resumed_from_bytes"], 128 * 1024)
                    self.assertFalse(os.path.exists(part_a_pull))
                    self.assertEqual(
                        inspect_gguf_file(path_b_pull)["fingerprint"],
                        inspect_gguf_file(dst_a_pull)["fingerprint"],
                    )

                    # 5. Re-run manifest check -> both models now Verified Match!
                    manifest_after = await fetch_peer_manifest("127.0.0.1", port_b, secret)
                    local_after = build_local_model_manifest([dir_a])
                    matrix = build_cluster_model_matrix(
                        local_after,
                        [{"id": "b", "node_name": "Node-B", "host": "127.0.0.1", "sync_port": port_b, "status": "online", "models": manifest_after["models"]}],
                    )
                    for row in matrix["models"]:
                        self.assertEqual(row["overall_status"], "ready")
                finally:
                    await srv_b.stop()

        asyncio.run(_scenario())


if __name__ == "__main__":
    unittest.main()
