from pygenn.genn_model import create_custom_neuron_class, GeNNModel

loop_bug = create_custom_neuron_class(
    "loop_bug",
    sim_code="""

    // Print $(id) outside the loop
    std::cout << "Outside loop $(id): " << $(id) << std::endl;
    
    // Print $(id) inside the loop
    for (int ii=5; ii<8; ii++){
        std::cout << "Inside  loop $(id): " << $(id)<< std::endl;
    }
    std::cout << std::endl;
    """,
)

model = GeNNModel("double", "loop_bug")
model.dT = 0.001
pop = model.add_neuron_population(
    pop_name="pop",
    num_neurons=2,
    neuron=loop_bug,
    param_space={},
    var_space={},
)

model.build()
model.load()

for i in range(1):
    model.step_time()
