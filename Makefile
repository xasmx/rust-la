RUSTC?=rustc
RUSTFLAGS=
SRC_DIR=src
RUST_SRC=${SRC_DIR}/lib.rs
BUILD_DIR=out
DOCS_DIR=doc

.PHONY: all
all: build docs

build: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) --out-dir $(BUILD_DIR) --crate-type lib $(RUST_SRC)

test-compile: $(RUST_SRC)
	mkdir -p $(BUILD_DIR)
	$(RUSTC) --test --out-dir $(BUILD_DIR) $(RUST_SRC)

.PHONY: test
test: test-compile $(RUST_SRC)
	RUST_TEST_TASKS=1 $(BUILD_DIR)/la

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(DOCS_DIR)

.PHONY: docs
docs:
	rustdoc -o $(DOCS_DIR) -L lib $(RUST_SRC)

