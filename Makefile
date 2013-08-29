RUSTC?=rustc
RUSTFLAGS=
SRC_DIR=src
RUST_SRC=${SRC_DIR}/la.rs
BUILD_DIR=out

.PHONY: all
all: build

build: $(RUST_SRC)
	-mkdir $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) --out-dir $(BUILD_DIR) --lib $(RUST_SRC)

test-compile: $(RUST_SRC)
	-mkdir $(BUILD_DIR)
	$(RUSTC) --test --out-dir $(BUILD_DIR) $(RUST_SRC)

.PHONY: test
test: test-compile $(RUST_SRC)
	RUST_THREADS=1 $(BUILD_DIR)/la

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)


