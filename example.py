"""Example usage script"""

import borg_parser


def main():
    # Setup
    path_to_runtime = borg_parser.datasets.water_energy()
    decision_names = ['w_with', 'w_con', 'w_emit']
    objective_names = [
        'f_gen',
        'f_with_tot',
        'f_con_tot',
        'f_disvi_tot',
        'f_emit',
        'f_ENS',
        'f_w_with',
        'f_w_con',
        'f_w_emit',
    ]
    metric_names = [
        'coal_output',
        'ng_output',
        'nuclear_output',
        'wind_output',
        'RI_output',
        'OC_output',
        'RC_output',
        'No Cooling System_output',
    ]

    # Create runtime object
    runtime = borg_parser.BorgRuntimeDiagnostic(
        path_to_runtime,
        n_decisions=len(decision_names),
        n_objectives=len(objective_names),
        n_metrics=len(metric_names),
    )
    runtime.set_decision_names(decision_names)
    runtime.set_objective_names(objective_names)
    runtime.set_metric_names(metric_names)

    # Interactive parallel
    exp = runtime.plot_interactive_front()
    exp.to_html('my_front.html')

    # Improvements
    fig = runtime.plot_improvements()
    fig.savefig('my_improvements.jpg')


if __name__ == '__main__':
    main()