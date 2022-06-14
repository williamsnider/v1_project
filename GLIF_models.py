from pygenn.genn_model import GeNNModel, create_custom_neuron_class

GLIF1 = create_custom_neuron_class(
    "GLIF1",
    param_names = ["C", "G", "El", "V_thres"],
    var_name_types=[("V", "scalar")],
    sim_code='$(V)+=1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;',
    threshold_condition_code='$(V)>=$(V_thres)',
    reset_code = '$(V)=$(El);',
)

GLIF2 = create_custom_neuron_class(
    "GLIF2",
    param_names = ["C", "G", "El", "bs"],
    var_name_types=[("V", "scalar"),("V_thres", "scalar") ],
    sim_code="""
    $(V)+=1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;
    $(th_s)+=-$(b_s)*$(th_s)*DT;
    """,
    threshold_condition_code='$(V) > $(th_inf) + $(th_s)',
    reset_code="""
    $(th_s) += $(delta) * $(th_s);
    $(V)= $(El) + $(fv) * ($(V) - $(El)) - $(delta)*$(V);
    """


)