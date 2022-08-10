import pygenn.genn_model

pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="$(Isyn) += $(x);",
    var_name_types=[("x", "scalar")],
    param_names=("tau"),
)

# pygenn.genn_model.create_custom_neuron_class()

model = pygenn.genn_model.GeNNModel()

model.add_synapse_population()
