{
    steps: {
        make: {
            type: "hf_smoke::make",
            n: 5,
        },
        total: {
            type: "hf_smoke::total",
            numbers: { type: "ref", ref: "make" },
        },
    },
}
