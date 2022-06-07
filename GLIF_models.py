from pygenn.genn_model import create_custom_neuron_class

GLIF_1 = create_custom_neuron_class(
    "GLIF",
    param_names = ["C", "G", "El","Ie", "V_thres"],
    var_name_types=[("V", "scalar")],
    sim_code='$(V)+=1/$(C)*($(Ie)-$(G)*($(V)-$(El)));',
    threshold_condition_code='$(V)>=$(V_thres)',
    reset_code = '$(V)=$(El);'
)