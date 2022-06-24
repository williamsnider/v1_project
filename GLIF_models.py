from pygenn.genn_model import create_custom_neuron_class

GLIF1 = create_custom_neuron_class(
    "GLIF1",
    param_names=["C", "G", "El", "th_inf", "spike_cut_length"],
    var_name_types=[("V", "scalar"), ("refractory_count", "int")],
    sim_code="""
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
    threshold_condition_code="$(V)>=$(th_inf)",
    reset_code="""
    $(V)=$(El);
    $(refractory_count) = $(spike_cut_length);
    """,
    is_auto_refractory_required=False,
)

GLIF2 = create_custom_neuron_class(
    "GLIF2",
    param_names=[
        "C",
        "G",
        "El",
        "a",
        "b",
        "a_spike",
        "b_spike",
        "spike_cut_length",
        "th_inf",
    ],
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
    threshold_condition_code="$(V) > $(th_inf) + $(th_s)",
    reset_code="""
    $(V)= $(El) + $(a) * ($(V) - $(El)) + $(b);
    $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    $(refractory_count) = $(spike_cut_length);
    """,
)

# TODO: Add vectors better so that they are not hardcoded to have just two elements
GLIF3 = create_custom_neuron_class(
    "GLIF3",
    param_names=[
        "C",
        "G",
        "El",
        "spike_cut_length",
        "th_inf",
    ],
    var_name_types=[("V", "double"), ("refractory_count", "int")],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    
    // Refractory period dynamics
    if ($(refractory_count) > 0) {
        $(V) += 0.0;
        $(ASC)[0] += 0.0;
        $(ASC)[1] += 0.0;
        }

    // Normal dynamics
    else {
        // Voltage 
        double sum_of_ASC = $(ASC)[0] + $(ASC)[1];
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
        
        // ASCurrents
        $(ASC)[0] = $(ASC)[0] * exp(-$(k)[0]*DT);
        $(ASC)[1] = $(ASC)[1] * exp(-$(k)[1]*DT);
    }


    // Decrement refractory_count by 1; Do not decrement past -1
    if ($(refractory_count) > -1) {
        $(refractory_count) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=0;
    $(ASC)[0] = $(asc_amp_array)[0] + $(ASC)[0] * $(r)[0] * exp(-($(k)[0] * DT * $(spike_cut_length)));
    $(ASC)[1] = $(asc_amp_array)[1] + $(ASC)[1] * $(r)[1] * exp(-($(k)[1] * DT * $(spike_cut_length)));
    $(refractory_count) = $(spike_cut_length);
    """,
)

GLIF4 = create_custom_neuron_class(
    "GLIF4",
    param_names=[
        "C",
        "G",
        "El",
        "a",
        "b",
        "a_spike",
        "b_spike",
        "spike_cut_length",
        "th_inf",
    ],
    var_name_types=[
        ("V", "double"),
        ("refractory_count", "int"),
        ("th_s", "double"),
    ],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    
    // Refractory period dynamics
    if ($(refractory_count) > 0) {
        $(V) += 0.0;
        $(ASC)[0] += 0.0;
        $(ASC)[1] += 0.0;
        }

    // Normal dynamics
    else {
        // Voltage 
        double sum_of_ASC = $(ASC)[0] + $(ASC)[1];
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
        
        // ASCurrents
        $(ASC)[0] = $(ASC)[0] * exp(-$(k)[0]*DT);
        $(ASC)[1] = $(ASC)[1] * exp(-$(k)[1]*DT);
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
    threshold_condition_code="$(V) > $(th_inf) + $(th_s)",
    reset_code="""
    $(V)=0;
    $(ASC)[0] = $(asc_amp_array)[0] + $(ASC)[0] * $(r)[0] * exp(-($(k)[0] * DT * $(spike_cut_length)));
    $(ASC)[1] = $(asc_amp_array)[1] + $(ASC)[1] * $(r)[1] * exp(-($(k)[1] * DT * $(spike_cut_length)));
    $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    $(refractory_count) = $(spike_cut_length);
    """,
)
