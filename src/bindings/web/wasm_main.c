// C main keeps the Zig export live in emcc's link graph.  The page drives
// ca_wasm_pump() from requestAnimationFrame after the runtime starts.

extern void ca_wasm_main(void);

int main(void) {
    ca_wasm_main();
    return 0;
}
