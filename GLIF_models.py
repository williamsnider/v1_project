from pygenn.genn_model import create_custom_neuron_class

GLIF1 = create_custom_neuron_class(
    "GLIF1",
    param_names=["C", "G", "El", "th_inf", "spike_cut_length"],
    var_name_types=[("V", "scalar"), ("refractory_countdown", "int")],
    sim_code="""

    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V) += 1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;
    }

    // Decrement refractory_countdown; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) >= $(th_inf)",
    reset_code="""
    $(V) = $(El);
    $(refractory_countdown) = $(spike_cut_length);
    """,
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
    var_name_types=[
        ("V", "double"),
        ("refractory_countdown", "int"),
        ("th_s", "double"),
    ],
    sim_code="""
    
    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V) += 1/$(C)*($(Isyn)-$(G)*($(V)-$(El)))*DT;
    }

    // Threshold - spike component
    // a_spike added on last timestep of refractory period
    if ($(refractory_countdown) == 1) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf) + $(th_s)",
    reset_code="""
    $(V)= $(El) + $(a) * ($(V) - $(El)) + $(b);
    
    // Handle case where spike_cut_length == 0
    if ($(spike_cut_length) == 0) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    $(refractory_countdown) = $(spike_cut_length);
    """,
)

GLIF3 = create_custom_neuron_class(
    "GLIF3",
    param_names=[
        "C",
        "G",
        "El",
        "spike_cut_length",
        "th_inf",
        "ASC_length",
    ],
    var_name_types=[("V", "double"), ("refractory_countdown", "int")],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    
    double sum_of_ASC = 0.0;
    
    // Sum after spike currents
    for (int i=0; i<2; i++)
        sum_of_ASC += $(ASC)[i];

    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown) > 0) {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] += 0.0;
    }
    else {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] = $(ASC)[i] * exp(-$(k)[i]*DT);
    }

    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=0;
    for (int i=0; i<$(ASC_length); i++)
        $(ASC)[i] = $(asc_amp_array)[i] + $(ASC)[i] * $(r)[i] * exp(-($(k)[i] * DT * $(spike_cut_length)));
    $(refractory_countdown) = $(spike_cut_length);
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
        "ASC_length",
    ],
    var_name_types=[
        ("V", "double"),
        ("refractory_countdown", "int"),
        ("th_s", "double"),
    ],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    double sum_of_ASC = 0.0;
    
    // Sum after spike currents
    for (int i=0; i<$(ASC_length); i++)
        sum_of_ASC += $(ASC)[i];
    
    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown) > 0) {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] += 0.0;
    }
    else {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] = $(ASC)[i] * exp(-$(k)[i]*DT);
    }

    // Threshold - spike component
    // a_spike added on last timestep of refractory period
    if ($(refractory_countdown) == 1) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf) + $(th_s)",
    reset_code="""
    $(V)= $(El) + $(a) * ($(V) - $(El)) + $(b);
    for (int i=0; i<$(ASC_length); i++)
        $(ASC)[i] = $(asc_amp_array)[i] + $(ASC)[i] * $(r)[i] * exp(-($(k)[i] * DT * $(spike_cut_length)));
    
    // Handle case where spike_cut_length == 0
    if ($(spike_cut_length) == 0) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    $(refractory_countdown) = $(spike_cut_length);
    """,
)

GLIF5 = create_custom_neuron_class(
    "GLIF5",
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
        "a_voltage",
        "b_voltage",
        "ASC_length",
    ],
    var_name_types=[
        ("V", "double"),
        ("refractory_countdown", "int"),
        ("th_s", "double"),
        ("th_v", "double"),
    ],
    extra_global_params=[
        ("ASC", "scalar*"),
        ("k", "scalar*"),
        ("asc_amp_array", "scalar*"),
        ("r", "scalar*"),
    ],
    sim_code="""
    
    double sum_of_ASC = 0.0;
    double I = 0.0;
    double beta = 0.0;
    double phi = 0.0;

    // Sum after spike currents
    for (int i=0; i<$(ASC_length); i++)
        sum_of_ASC += $(ASC)[i];

    // Threshold - spike component
    // a_spike added on last timestep of refractory period
    if ($(refractory_countdown) == 1) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    // Threshold - voltage component
    // Must occur before ASC updated
    if ($(refractory_countdown) <= 0) {
        I = $(Isyn) + sum_of_ASC;
        beta = (I+$(G)*$(El))/$(G);
        phi = $(a_voltage)/($(b_voltage)-$(G)/$(C));
        $(th_v) = phi*($(V)-beta)*exp(-$(G)*DT/$(C))+1/(exp($(b_voltage)*DT))*($(th_v)-phi*($(V)-beta)-($(a_voltage)/$(b_voltage))*(beta-$(El))-0) +($(a_voltage)/$(b_voltage))*(beta-$(El));
    } else {
        $(th_v) += 0.0;
    }

    // Voltage
    if ($(refractory_countdown) > 0) {
        $(V) += 0.0;
    }
    else {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown) > 0) {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] += 0.0;
    }
    else {
        for (int i=0; i<$(ASC_length); i++)
            $(ASC)[i] = $(ASC)[i] * exp(-$(k)[i]*DT);
    }

    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf) + $(th_s) + $(th_v)",
    reset_code="""
    $(V)= $(El) + $(a) * ($(V) - $(El)) + $(b);
    $(th_v) = $(th_v);
    for (int i=0; i<$(ASC_length); i++)
        $(ASC)[i] = $(asc_amp_array)[i] + $(ASC)[i] * $(r)[i] * exp(-($(k)[i] * DT * $(spike_cut_length)));
    
    // Threshold - spike compoennt - handle case where spike_cut_length == 0
    if ($(spike_cut_length) == 0) {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT) + $(a_spike);
    }
    else {
        $(th_s) = $(th_s) * exp(-$(b_spike)*DT);
    }

    $(refractory_countdown) = $(spike_cut_length);
    """,
)
