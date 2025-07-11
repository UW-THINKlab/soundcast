# Copyright [2014] [Puget Sound Regional Council]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!python.exe
# PSRC SoundCast Model Runner
# ===========================

import os
import sys
import datetime
import subprocess
import json
import shutil
import random

sys.path.append(os.path.join(os.getcwd(), "inputs"))
sys.path.append(os.path.join(os.getcwd(), "scripts"))
from scripts import logcontroller
from scripts.settings import run_args
from scripts.settings import state
from scripts.settings import data_wrangling
from scripts.skimming import SkimsAndPaths
from scripts.network import network_importer
from scripts.accessibility import accessibility
from scripts.supplemental import create_ixxi_work_trips
from scripts.supplemental import generation
from scripts.supplemental import distribute_non_work_ixxi
from scripts.supplemental import create_airport_trips
from scripts.trucks import truck_model
from scripts.utils import update_parking
from scripts.utils import urbansim_skims
from scripts.summarize.standard import (
    daily_bank,
    network_summary,
    transit_summary,
    emissions,
    agg,
    validation,
    job_accessibility,
)

state = state.generate_state(run_args.args.configs_dir)


def accessibility_calcs():
    data_wrangling.copy_accessibility_files(state)
    create_ixxi_work_trips.main(state)
    
    if state.input_settings.base_year != state.input_settings.model_year:
        print("Update parcels with parking zone data")
        update_parking.main(state)

    accessibility.run(state)


@data_wrangling.timed
def build_seed_skims(max_iterations):
    print("Processing skims and paths.")
    time_copy = datetime.datetime.now()
    returncode = subprocess.call(
        [
            sys.executable,
            "scripts/skimming/SkimsAndPaths.py",
            str(max_iterations),
            state.input_settings.model_year,
            "-use_daysim_output_seed_trips",
        ]
    )
    if returncode != 0:
        sys.exit(1)

    time_skims = datetime.datetime.now()
    print("###### Finished skim building:", str(time_skims - time_copy))


def build_free_flow_skims(max_iterations):
    print("Building free flow skims.")
    time_copy = datetime.datetime.now()
    SkimsAndPaths.run(True, max_iterations)

    time_skims = datetime.datetime.now()
    print("###### Finished skim building:", str(time_skims - time_copy))


@data_wrangling.timed
def modify_config(config_vals):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]  # <-- absolute dir the script is in
    config_template_path = "daysim_configuration_template.properties"
    config_path = "Daysim/daysim_configuration.properties"

    abs_config_path_template = os.path.join(script_dir, config_template_path)
    abs_config_path_out = os.path.join(script_dir, config_path)
    print(abs_config_path_template)
    config_template = open(abs_config_path_template, "r")
    config = open(abs_config_path_out, "w")

    try:
        for line in config_template:
            for config_temp, config_update in config_vals:
                if config_temp in line:
                    line = line.replace(config_temp, str(config_update))
            config.write(line)

        config_template.close()
        config.close()

    except:
        config_template.close()
        config.close()
        print(" Error creating configuration template file")
        sys.exit(1)


@data_wrangling.timed
def build_shadow_only():
    for shad_iter in range(0, len(state.emme_settings.shadow_work)):
        modify_config(
            [
                ("$SHADOW_PRICE", "true"),
                ("$SAMPLE", state.emme_settings.shadow_work[shad_iter]),
                ("$RUN_ALL", "false"),
            ]
        )
        logger.info(
            "Start of%s iteration of work location for shadow prices", str(shad_iter)
        )
        returncode = subprocess.call(
            "Daysim/Daysim.exe -c Daysim/daysim_configuration.properties"
        )
        logger.info(
            "End of %s iteration of work location for shadow prices", str(shad_iter)
        )
        if returncode != 0:
            sys.exit(1)
        returncode = subprocess.call(
            [sys.executable, "scripts/utils/shadow_pricing_check.py"]
        )
        shadow_con_file = open("outputs/shadow_rmse.txt", "r")
        rmse_list = shadow_con_file.readlines()
        iteration_number = len(rmse_list)
        current_rmse = float(rmse_list[iteration_number - 1].rstrip("\n"))
        if current_rmse < state.emme_settings.shadow_con:
            print("done with shadow prices")
            shadow_con_file.close()
            return


def run_truck_supplemental(iteration, statwe):
    if state.input_settings.run_supplemental_trips:
        if iteration == 0:
            generation.main(state)
        distribute_non_work_ixxi.main(state)
        create_airport_trips.main(state)

    if state.input_settings.run_truck_model:
        truck_model.main(state)


@data_wrangling.timed
def daysim_assignment(iteration):
    ########################################
    # Run Daysim Activity Models
    ########################################

    if state.input_settings.run_daysim:
        logger.info("Start of %s iteration of Daysim", str(iteration))
        returncode = subprocess.call(
            "Daysim/Daysim.exe -c Daysim/daysim_configuration.properties"
        )
        logger.info("End of %s iteration of Daysim", str(iteration))
        if returncode != 0:
            sys.exit(1)

    ########################################
    # Calcualte Trucks and Supplemental Demand
    ########################################
    run_truck_supplemental(iteration, state.input_settings)

    ########################################
    # Assign Demand to Networks
    ########################################

    if state.input_settings.run_skims_and_paths:
        logger.info("Start of iteration %s of Skims and Paths", str(iteration))
        num_iterations = str(state.emme_settings.max_iterations_list[iteration])
        SkimsAndPaths.run(False, num_iterations)
        logger.info("End of iteration %s of Skims and Paths", str(iteration))


@data_wrangling.timed
def check_convergence(iteration):
    converge = "not yet"
    if (
        iteration > 0
        and state.emme_settings.pop_sample[iteration]
        <= state.emme_settings.min_pop_sample_convergence_test
    ):
        con_file = open("outputs/logs/converge.txt", "r")
        converge = json.load(con_file)
        con_file.close()
    return converge


@data_wrangling.timed
def run_all_summaries():
    daily_bank.main(state)
    network_summary.main(state)
    transit_summary.main(state)
    emissions.main(state)
    agg.main(state)
    validation.main(state)
    job_accessibility.main(state)
    subprocess.run(
        "conda activate summary && python scripts/summarize/create_quarto_notebooks.py && conda deactivate",
        shell=True,
    )


def get_current_commit_hash():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit = "0000000"
    return commit

def check_settings(state):

    # Check that distance pricing has b1een applied correctly
    if state.input_settings.model_year != state.input_settings.base_year and not state.input_settings.add_distance_pricing:
        # get user input y/n to continue
        print(
            "Distance pricing is not applied to this future year run. "
            "Do you want to continue? (y/n)"
        )
        user_input = input().strip().lower()
        if user_input != "y":
            print("Exiting the model run.")
            sys.exit(1)

    # If banks exist, don't erase them without confirmation
    if state.input_settings.run_setup_emme_bank_folders and os.path.isdir('Banks'):
        # get user input y/n to continue
        print(
            "Banks will be deleted and overwritten. "
            "Do you want to continue? (y/n)"
        )
        user_input = input().strip().lower()
        if user_input != "y":
            print("Exiting the model run.")
            sys.exit(1)


def main():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    ########################################
    # Initialize Banks, Projects, Directories
    ########################################

    check_settings(state)

    hash = get_current_commit_hash()
    logger.info("Using Git hash %s ", str(hash))

    data_wrangling.store_settings(state)
    data_wrangling.build_output_dirs()
    data_wrangling.update_daysim_modes(state)
    data_wrangling.update_skim_parameters(state)

    # this import statement needs to happen here, after update_skim_parameters:

    if state.input_settings.run_setup_emme_bank_folders:
        data_wrangling.setup_emme_bank_folders(state)

    if state.input_settings.run_setup_emme_project_folders:
        data_wrangling.setup_emme_project_folders(state)

    if state.input_settings.run_copy_scenario_inputs:
        data_wrangling.copy_scenario_inputs(state)

    if state.input_settings.run_integrated:
        data_wrangling.import_integrated_inputs(state)

    # If debugging SkimsAndPaths, cannot have another instance of Emme Modeller running
    if not state.input_settings.debug_skims_and_paths:
        # This is the main project/modeller instance that will stay open for the entire run.
        # Used for all single process EMME workflows.
        state.create_main_project()

    if state.input_settings.run_accessibility_calcs:
        accessibility_calcs()

    if not os.path.exists("working"):
        os.makedirs("working")

    ########################################
    # Initialize Networks
    ########################################

    if state.input_settings.run_import_networks:
        time_copy = datetime.datetime.now()
        logger.info("Start of network importer")
        network_importer.run_importer(state)
        logger.info("End of network importer")
        time_network = datetime.datetime.now()

    ########################################
    # Start with Free Flow Skims
    ########################################

    if state.input_settings.run_skims_and_paths_free_flow:
        build_free_flow_skims(10)

    ########################################
    # Generate Demand and Assign Volumes
    # Main Loop
    ########################################

    if (
        state.input_settings.run_daysim
        or state.input_settings.run_skims_and_paths
        or state.input_settings.run_supplemental_trips
        or state.input_settings.run_truck_model
    ):
        for iteration in range(len(state.emme_settings.pop_sample)):
            print("We're on iteration %d" % (iteration))
            logger.info(("We're on iteration %d\r\n" % (iteration)))
            time_start = datetime.datetime.now()
            logger.info("Starting run at %s" % str((time_start)))

            # set daysim model inputs (allows for daysim to be run with different configurations)

            if not state.input_settings.should_build_shadow_price:
                if state.input_settings.include_telecommute:
                    telecommute = "true"
                else:
                    telecommute = "false"
                # Set up your Daysim Configration
                modify_config(
                    [
                        ("$SHADOW_PRICE", "true"),
                        ("$SAMPLE", state.emme_settings.pop_sample[iteration]),
                        ("$RUN_ALL", "true"),
                        ("$TELECOMMUTE", telecommute),
                        ("$INPUTDIR", state.input_settings.abm_model),
                    ]
                )

            else:
                # We are building shadow prices from scratch, only use shadow pricing if pop sample is 2 or less
                if state.emme_settings.pop_sample[iteration - 1] > 2:
                    modify_config(
                        [
                            ("$SHADOW_PRICE", "false"),
                            ("$SAMPLE", state.emme_settings.pop_sample[iteration]),
                            ("$RUN_ALL", "true"),
                        ]
                    )
                else:
                    modify_config(
                        [
                            ("$SHADOW_PRICE", "true"),
                            ("$SAMPLE", state.emme_settings.pop_sample[iteration]),
                            ("$RUN_ALL", "true"),
                        ]
                    )

            # Run Skimming and/or Daysim
            daysim_assignment(iteration)

            # Check Convergence
            converge = check_convergence(iteration)
            if converge == "stop":
                print("System converged!")
                break
            print(
                "The system is not yet converged. Daysim and Assignment will be re-run."
            )

    # If building shadow prices, update work and school shadow prices
    # using converged skims from current run, then re-run daysim and assignment.
    if state.input_settings.should_build_shadow_price:
        build_shadow_only()
        modify_config(
            [("$SHADOW_PRICE", "true"), ("$SAMPLE", "1"), ("$RUN_ALL", "true")]
        )
        # This function needs an iteration parameter. Value of 1 is fine.
        daysim_assignment(1)

    # Export skims for use in Urbansim if needed
    if state.input_settings.run_integrated:
        urbansim_skims.main(state)

    if state.input_settings.run_summaries:
        run_all_summaries()

    data_wrangling.clean_up()
    print(
        "###### OH HAPPY DAY!  ALL DONE. GO GET "
        + random.choice(state.emme_settings.good_thing)
    )


if __name__ == "__main__":
    # Set up logging
    logger = logcontroller.setup_custom_logger("main_logger", r"soundcast_log.txt")

    logger.info("--------------------NEW RUN STARTING--------------------")
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()
    elapsed_total = end_time - start_time
    logger.info("--------------------RUN ENDING--------------------")
    logger.info("TOTAL RUN TIME %s" % str(elapsed_total))

    if state.input_settings.delete_banks:
        shutil.rmtree("/Banks", ignore_errors=True)
