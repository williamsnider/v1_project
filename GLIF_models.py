from pygenn.genn_model import create_custom_neuron_class

GLIF1 = create_custom_neuron_class(
    "GLIF1",
    param_names = ["C", "G", "El", "th_inf", "spike_cut_length"],
    var_name_types=[("V", "scalar"), ("refractory_count", "int")],
    sim_code=
    """
    if ($(refractory_count) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;
        }

    // Decrement refractory_count; Do not decrement past -1
    if ($(refractory_count) != -1) {
        $(refractory_count) -= 1;
    }
    """,
    threshold_condition_code='$(V)>=$(th_inf)',
    reset_code = 
    """
    $(V)=$(El);
    $(refractory_count) = $(spike_cut_length);
    """,
    is_auto_refractory_required=False,
)

GLIF2 = create_custom_neuron_class(
    "GLIF2",
    param_names = ["C", "G", "El", "a", "b","a_spike", "b_spike", "spike_cut_length", "th_inf"],
    var_name_types=[("V", "double"), ("refractory_count", "int"), ("th_s", "double")],
    sim_code="""
    
    // Voltage
    if ($(refractory_count) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;
    }

    // Spike component of threshold
    if ($(refractory_count) == 1) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    // Decrement refractory_count by 1; Do not decrement past -1
    if ($(refractory_count) > -1) {
        $(refractory_count) -= 1;
    }
    """,
    threshold_condition_code='$(V) > $(th_inf) + $(th_s)',
    reset_code="""
    $(V)= $(El) + $(a) * ($(V) - $(El)) + $(b);
    $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    $(refractory_count) = $(spike_cut_length);
    """


)