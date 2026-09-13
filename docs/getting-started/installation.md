# Installation & Setup

This page guides you through installing **RiftPoint** and setting up your Python environment.

---

## 📋 Prerequisites

- **Python:** `>= 3.11` (Python 3.11 or 3.12 recommended)
- **Operating System:** Linux, macOS (Apple Silicon / Intel), or Windows
- **LangGraph:** `>= 0.2.0`
- **Rust Backend:** `janus-tachyon-rs >= 0.2.1` (automatically installed as a binary wheel)

---

## 📦 Package Installation

### With `uv` (Recommended)

```bash
uv add riftpoint
```

### With `pip`

```bash
pip install riftpoint
```

### With `Poetry`

```bash
poetry add riftpoint
```

---

## 🛠️ Optional Dependencies

RiftPoint provides optional extras for extended plotting and visualization capabilities:

```bash
# Install with Matplotlib DAG rendering support
uv add "riftpoint[plot]"
# Or with pip:
pip install "riftpoint[plot]"
```

---

## 🔍 Verification

Verify your installation by running a simple smoke test in your terminal:

```bash
python -c "import riftpoint; print('RiftPoint version:', riftpoint.__version__)"
```

If successful, you are ready to proceed to the [Quickstart Guide](quickstart.md).
