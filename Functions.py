
import os
import time
import numpy as np
import pyomo.environ as pe
from model_serializer import StoreSpec, from_json, to_json
from pyomo.common.errors import InfeasibleConstraintException
from math import fabs
from pyomo.contrib.fbbt.fbbt import fbbt


from pyomo.opt.base.solvers import SolverFactory


def run_function_dbd_scheduling_cost_min_ref_2(model_fun_feas,minimum_obj,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):

    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':0.01}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                sub_options_feasibility={}
                sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > cplex.opt \n','intsollim 1\n','mipemphasis 0\n','$offecho \n']}
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy)#minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k-1)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=1e-5: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                    final_sol=[]
                    for I_J in m_solution.I_J:
                            final_sol.append(pe.value(m_solution.Nref[I_J])+1)    
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})
                    m_return= m_solution                     
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            _cost[1]=10

            _cost[2]=15
            _cost[3]=30

            _cost[4]=5
            _cost[5]=25

            _cost[6]=5
            _cost[7]=20

            _cost[8]=20
            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k-1)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
    return important_info,D,x_actual,m_return

def run_function_dbd_scheduling_cost_min_nonlinear_ref_2(model_fun_feas,minimum_obj,absolute_gap,epsilon,initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random: bool=False,use_multi_start: bool=False,n_points_multstart: int=10,sub_solver_opt: dict={}, tee: bool=False, known_solutions: dict={}):

    #------------------------------------------PARAMETER INITIALIZATION---------------------------------------------------------------
    important_info={}
    iterations=range(1,maxiter+1)
    D_random={}
    initial_Stage=3 #stage where the algorithm will be initialized: 1 is feasibility1, 2 is feasibility2 and 3 is optimality
    #------------------------------------------REFORMULATION WITH EXTERNAL VARIABLES--------------------------------------------------
    model = model_fun(**kwargs)
    _, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=False) 
    #------------------------------------------PRE PROCESSING-------------------------------------------------------------------------
    start=time.time()
    #-----------------------------------D-BD ALGORITHM-----------------------------------------------------------------------
    if initial_Stage==1 or initial_Stage==2 or initial_Stage==3:
        if tee==True:
            print('stage 3...')
        if initial_Stage==3:
            x_actual=initialization
            D={}
            D=D_random.copy()
        D.update(known_solutions) # updating dictionary with previously evaluated solutions
        x_dict={}  #value of x at each iteration
        fobj_actual=infinity_val
        start = time.time()

        sub_options_feasibility={}
        if nlp_solver=='dicopt': #TODO: CONOPT 4 used for a specific case study
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > dicopt.opt \n','feaspump 2\n','MAXCYCLES 1\n','stop 0\n','fp_sollimit 1\n','$offecho \n']}
        elif nlp_solver=='baron':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > baron.opt \n','FirstFeas 1\n',' NumSol 1\n','$offecho \n']}
        elif nlp_solver=='lindoglobal':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > lindoglobal.opt \n',' GOP_OPT_MODE 0\n','$offecho \n']}
        elif nlp_solver=='antigone':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > antigone.opt \n','abs_opt_tol 100\n','rel_opt_tol 1\n','$offecho \n']}
        elif nlp_solver=='sbb':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > sbb.opt \n','intsollim 1\n','$offecho \n']}                
        elif nlp_solver=='bonmin':
            sub_options_feasibility={'add_options':['GAMS_MODEL.optfile = 1;','\n','$onecho > bonmin.opt \n','bonmin.pump_for_minlp yes\n','pump_for_minlp.solution_limit 1\n','solution_limit 1\n','$offecho \n']}   
        for k in iterations:
            #print(k)
            #if first iteration, initialize
            #if k==1:
            #    x_actual=initialization
            #print(x_actual)
            #update current value of x in the dictionary
            x_dict[k]=x_actual
            if tee==True and k==1:
                print('S3---- User provided lower bound= '+str(minimum_obj))
            #print(x_actual)
            #calculate objective function for current point and its neighborhood (subproblem)
            if k!=1:
                kwargs_Feas={'objective':minimum_obj,'epsilon':absolute_gap}# TODO: use this epsilon as input
                m_feas=model_fun_feas(**kwargs_Feas)
                pe.TransformationFactory('core.logical_to_linear').apply_to(m_feas)
                m_solution=solve_subproblem(m_feas,subproblem_solver = nlp_solver,subproblem_solver_options= sub_options_feasibility,timelimit= 1000000,gams_output = False,tee= False,rel_tol = 0)
                
                if m_solution.dsda_status=='Optimal':
                    fobj_actual=pe.value(m_solution.obj_dummy) #minimum_obj #I should extract the solution of the subproblem, but I know that this is going to be its solution
                else:
                    fobj_actual=infinity_val
                if tee==True:
                    print('S3--'+'--iter '+str(k-1)+'---  |  '+'ext. vars= '+str(x_actual)+'   |   sub. obj= '+str(fobj_actual))
                #Add points to D
                D.update({tuple(x_actual):fobj_actual})
                #print(D)
                #Calculate new convex hull and dd cuts to the current model
                #define model
                if fabs(fobj_actual-minimum_obj)<=absolute_gap: #or all(fobj_actual<=val for val in D.values()): # if minimum over D, then it is minimum over neighborhood, plus I guarantee that no other neighbor has a better solution 
                #if all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if [pe.value(m.x1),pe.value(m.x2)]==x_actual and all(list(new_values.values())[0]<=val for val in list(new_values.values())[1:]):
                #if
                
                    final_sol=[]
                    for I_J in m_solution.I_J:
                            final_sol.append(pe.value(m_solution.Nref[I_J])+1)     
                    x_actual=final_sol
                    D.update({tuple(final_sol):fobj_actual})   
                    m_return= m_solution 
                    actual_absolute_gap= fabs(fobj_actual-minimum_obj) 
                    actual_relative_gap=(actual_absolute_gap/fabs(minimum_obj))*100              
                    break
            m,_=build_master(number_of_external_variables,lower_bounds,upper_bounds,x_actual,3,D)            
            # for i in x_dict:
            #     cuts=convex_clousure(D,x_dict[i])
            #     #print(cuts)
            #     m.cuts.add(sum(m.x[posit]*float(cuts[posit-1]) for posit in m.extset)+float(cuts[-1])<=m.zobj)
                #m.cuts.add(m.x1*float(cuts[0])+m.x2*float(cuts[1])+float(cuts[2])<=m.zobj)
            _cost={}# TODO: GENERALIZE THIS!!!!
            _cost[1]=10

            _cost[2]=15
            _cost[3]=30

            _cost[4]=5
            _cost[5]=25

            _cost[6]=5
            _cost[7]=20

            _cost[8]=20


            if k==1:
                m.cuts.add(minimum_obj<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset))  
            else:
                m.cuts.add(minimum_obj+epsilon<=sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)) #TODO: epsilon must be the minimum coefficient in the objective function
            m.cuts.add(sum(_cost[posit]*(m.x[posit]-1) for posit in m.extset)<=m.zobj)           
            #Solve master problem       
            SolverFactory('gams', solver='cplex').solve(m, tee=False)
            if tee==True:
                print('S3--'+'--iter '+str(k-1)+'---   |   master. obj= '+str(pe.value(m.zobj)))
            #Stop?
            #print([pe.value(m.x1),pe.value(m.x2)])
            #print(new_values)

            x_actual=[round(pe.value(m.x[posita])) for posita in m.extset]
            minimum_obj=pe.value(m.zobj)
        end = time.time()
        #print('stage 3: method_3 time:',end - start,'method_3 obj:',D[tuple(x_actual)])
        #print('Cuts calculated from the central points evaluated so far.')
        #print(x_dict,'\n')
        important_info['m3_s3']=[D[tuple(x_actual)],end - start,'if objective in m1_s2 is 0-> solution is feasible and optimal']
        if tee==True:
            print('-------------------------------------------')
            print('Best objective= '+str(D[tuple(x_actual)])+'   |   CPU time [s]= '+str(end-start)+'   |   ext. vars='+str(x_actual))
            print('optca=',str(actual_absolute_gap),'| optcr=',str(actual_relative_gap),'%')
    return important_info,D,x_actual,m_return

def get_external_information(
    m: pe.ConcreteModel(),
    ext_ref,
    tee: bool = False,
):
    """
    Function that obtains information from the model to perform the reformulation with external variables.
    The model must be a GDP problem with exactly one "Exactly(k_j, [Y_j1,Y_j2,Y_j3,...])" constraint for each list of variables
    [Y_j1,Y_j2,Y_j3,...] that is going to be reformulated over set j.
    Args:
        m: GDP model that is going to be reformulated
        ext_ref: Dictionary with Boolean variables to be reformulated (keys) and their corresponding ordered sets (values). Both keys and values are pyomo objects.
        tee: Display reformulation
    Returns:
        reformulation_dict: A dictionary of dictionaries that looks as follows:
            {1:{'exactly_number':Number of external variables for this type,
                'Boolean_vars_names':list with names of the ordered Boolean variables to be reformulated,
                'Boolean_vars_ordered_index': Indexes where the external reformulation is applied,
                'Ext_var_lower_bound': Lower bound for this type of external variable,
                'Ext_var_upper_bound': Upper bound for this type of external variable },
             2:{...},...}

            The first key (positive integer) represent a type of external variable identified in the model. For this type of external variable
            a dictionary is created.
        number_of_external_variables: Number of external variables
        lower_bounds: Dictionary with positive integer keys identifying the external variable, and its lower bound as value
        upper_bounds: Dictionary with positive integer keys identifying the external variable, and its upper bound as value

    """

    # If Boolean variables that are going to be reformulated are defined over multiple sets try:
    try:
        # index of the set where reformultion can be applied for a given boolean variable
        ref_index = {}
        # index of the sets where the reformulation cannot be applied for a given boolean variable
        no_ref_index = {}
        for i in ext_ref:
            ref_index[i] = []
            no_ref_index[i] = []
            for index_set in range(len(i.index_set()._sets)):
                if i.index_set()._sets[index_set].name == ext_ref[i].name:
                    ref_index[i].append(index_set)
                else:
                    no_ref_index[i].append(index_set)
    # If boolean variables that are going to be reformulated are defined over a single set except:
    except:
        # index of the set where reformultion can be applied for a given boolean variable
        ref_index = {}
        # index of the sets where the reformulation cannot be applied for a given boolean variable
        no_ref_index = {}
        for i in ext_ref:
            ref_index[i] = []
            no_ref_index[i] = []
            if i.index_set().name == ext_ref[i].name:
                ref_index[i].append(0)
            else:
                no_ref_index[i].append(0)

    # Identify the variables that can be reformulated by performing a loop over logical constraints
    count = 1
    # dict of dicts: it contains information from the exactly variables that can be reformulated into external variables.
    reformulation_dict = {}
    for c in m.component_data_objects(pe.LogicalConstraint, descend_into=True):
        if c.body.getname() == 'exactly':
            exactly_number = c.body.args[0]
            for possible_Boolean in ext_ref:

                # expected boolean variable where the reformulation is going to be applied
                expected_Boolean = possible_Boolean.name
                Boolean_name_list = []
                Boolean_name_list = Boolean_name_list + \
                    [c.body.args[1:][k]._component()._name for k in range(
                        len(c.body.args[1:]))]
                if all(x == expected_Boolean for x in Boolean_name_list):
                    # expected ordered set index where the reformulation is going to be applied
                    expected_ordered_set_index = ref_index[possible_Boolean]
                    # index of sets where the reformulation is not applied
                    index_of_other_sets = no_ref_index[possible_Boolean]
                    if len(index_of_other_sets) >= 1:  # If there are other indexes
                        Other_Sets_listOFlists = []
                        verification_Other_Sets_listOFlists = []
                        for j in index_of_other_sets:
                            Other_Sets_listOFlists.append(
                                [c.body.args[1:][k].index()[j] for k in range(len(c.body.args[1:]))])
                            if all(c.body.args[1:][x].index()[j] == c.body.args[1:][0].index()[j] for x in range(len(c.body.args[1:]))):
                                verification_Other_Sets_listOFlists.append(
                                    True)
                            else:
                                verification_Other_Sets_listOFlists.append(
                                    False)
                        # If we get to this point and it is true, it means that we can apply the reformulation for this combination of Boolean var and Exactly-type constraint
                        if all(verification_Other_Sets_listOFlists):
                            reformulation_dict[count] = {}
                            reformulation_dict[count]['exactly_number'] = exactly_number
                            # rearange boolean vars in constraint
                            sorted_args = sorted(c.body.args[1:], key=lambda x: x.index()[
                                                 expected_ordered_set_index[0]])
                            # Now work with the ordered version sorted_args instead of c.body.args[1:]
                            reformulation_dict[count]['Boolean_vars_names'] = [
                                sorted_args[k].name for k in range(len(sorted_args))]
                            reformulation_dict[count]['Boolean_vars_ordered_index'] = [sorted_args[k].index(
                            )[expected_ordered_set_index[0]] for k in range(len(sorted_args))]
                            reformulation_dict[count]['Ext_var_lower_bound'] = 1
                            reformulation_dict[count]['Ext_var_upper_bound'] = len(
                                sorted_args)

                            count = count+1
                    # If there is only one index, then we can apply the reformulation at this point
                    else:
                        reformulation_dict[count] = {}
                        reformulation_dict[count]['exactly_number'] = exactly_number
                        # rearange boolean vars in constraint
                        sorted_args = sorted(
                            c.body.args[1:], key=lambda x: x.index())
                        # Now work with the ordered version sorted_args instead of c.body.args[1:]
                        reformulation_dict[count]['Boolean_vars_names'] = [
                            sorted_args[k].name for k in range(len(sorted_args))]
                        reformulation_dict[count]['Boolean_vars_ordered_index'] = [
                            sorted_args[k].index() for k in range(len(sorted_args))]
                        reformulation_dict[count]['Ext_var_lower_bound'] = 1
                        reformulation_dict[count]['Ext_var_upper_bound'] = len(
                            sorted_args)

                        count = count+1

    number_of_external_variables = sum(
        reformulation_dict[j]['exactly_number'] for j in reformulation_dict)

    lower_bounds = {}
    upper_bounds = {}

    exvar_num = 1
    for i in reformulation_dict:
        for j in range(reformulation_dict[i]['exactly_number']):
            lower_bounds[exvar_num] = reformulation_dict[i]['Ext_var_lower_bound']
            upper_bounds[exvar_num] = reformulation_dict[i]['Ext_var_upper_bound']
        exvar_num = exvar_num+1

    if tee:
        print('\nReformulation Summary\n--------------------------------------------------------------------------')
        exvar_num = 0
        for i in reformulation_dict:
            for j in range(reformulation_dict[i]['exactly_number']):
                print('External variable x['+str(exvar_num)+'] '+' is associated to '+str(reformulation_dict[i]['Boolean_vars_names']) +
                      ' and it must be within '+str(reformulation_dict[i]['Ext_var_lower_bound'])+' and '+str(reformulation_dict[i]['Ext_var_upper_bound'])+'.')
                exvar_num = exvar_num+1

        print('\nThere are '+str(number_of_external_variables) +
              ' external variables in total')

    return reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds

def solve_subproblem(
    m: pe.ConcreteModel(),
    subproblem_solver: str = 'knitro',
    subproblem_solver_options: dict = {},
    timelimit: float = 1000,
    gams_output: bool = False,
    tee: bool = False,
    rel_tol: float = 0,
) -> pe.ConcreteModel():
    """
    Function that checks feasibility and optimizes subproblem model.
    Note integer variables have to be previously fixed in the external reformulation
    Args:
        m: Fixed subproblem model that is to be solved
        subproblem_solver: MINLP or NLP solver algorithm
        timelimit: time limit in seconds for the solve statement
        gams_output: Determine keeping or not GAMS files
        tee: Display iteration output
        rel_tol: Relative optimality tolerance
    Returns:
        m: Solved subproblem model
    """
    # Initialize D-SDA status
    m.dsda_status = 'Initialized'
    m.dsda_usertime = 0
    start_prep=time.time()
    try:
        # Feasibility and preprocessing checks
        preprocess_problem(m, simple=True)

    except InfeasibleConstraintException:
        m.dsda_status = 'FBBT_Infeasible'
        return m
    end_prep=time.time()
    m.dsda_usertime =m.dsda_usertime + (end_prep-start_prep)
    output_options = {}

    # Output report
    if gams_output:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        gams_path = os.path.join(dir_path, "gamsfiles/")
        if not(os.path.exists(gams_path)):
            print('Directory for automatically generated files ' +
                  gams_path + ' does not exist. We will create it')
            os.makedirs(gams_path)
        output_options = {'keepfiles': True,
                          'tmpdir': gams_path,
                          'symbolic_solver_labels': True}

    subproblem_solver_options['add_options'] = subproblem_solver_options.get(
        'add_options', [])
    subproblem_solver_options['add_options'].append(
        'option reslim=%s;' % timelimit)
    subproblem_solver_options['add_options'].append(
        'option optcr=%s;' % rel_tol)
    # Solve
    solvername = 'gams'
    
    if subproblem_solver=='OCTERACT':
        opt = SolverFactory(solvername)
    else:
        opt = SolverFactory(solvername, solver=subproblem_solver)

    m.results = opt.solve(m, tee=tee,
                          **output_options,
                          **subproblem_solver_options,
                          skip_trivial_constraints=True,
                          )

    m.dsda_usertime =m.dsda_usertime + m.results.solver.user_time

    # Assign D-SDA status
    if m.results.solver.termination_condition == 'infeasible' or m.results.solver.termination_condition == 'other' or m.results.solver.termination_condition == 'unbounded' or m.results.solver.termination_condition == 'invalidProblem' or m.results.solver.termination_condition == 'solverFailure' or m.results.solver.termination_condition == 'internalSolverError' or m.results.solver.termination_condition == 'error'  or m.results.solver.termination_condition == 'resourceInterrupt' or m.results.solver.termination_condition == 'licensingProblem' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'noSolution' or m.results.solver.termination_condition == 'intermediateNonInteger':
        m.dsda_status = 'Evaluated_Infeasible'
    else:  # Considering locallyOptimal, optimal, globallyOptimal, and maxtime TODO Fix this
        m.dsda_status = 'Optimal'
    # if m.results.solver.termination_condition == 'locallyOptimal' or m.results.solver.termination_condition == 'optimal' or m.results.solver.termination_condition == 'globallyOptimal':
    #     m.dsda_status = 'Optimal'

    return m

def initialize_model(
    m: pe.ConcreteModel(),
    json_path=None,
    from_feasible: bool = False,
    feasible_model: str = '',
) -> pe.ConcreteModel():
    """
    Function that return an initialized model from an existing json file
    Args:
        m: Pyomo model that is to be initialized
        from_feasible: If initialization is made from an external file
        feasible_model: Feasible initialization path or example
    Returns:
        m: Initialized Pyomo model
    """

    wts = StoreSpec.value()

    if json_path is None:
        os.path.join(os.path.curdir)

        dir_path = os.path.dirname(os.path.abspath(__file__))

        if from_feasible:
            json_path = os.path.join(
                dir_path, feasible_model+'_initialization.json')
        else:
            json_path = os.path.join(
                dir_path, 'dsda_initialization.json')

    from_json(m, fname=json_path, wts=wts)
    return m

def generate_initialization(
    m: pe.ConcreteModel(),
    starting_initialization: bool = False,
    model_name: str = '',
    human_read: bool = True,
    wts=StoreSpec.value(),
):
    """
    Function that creates a json file for initialization based on a model m
    Args:
        m: Base Pyomo model for initializtion
        starting_intialization: Use to create "dsda_starting_initialization.json" file with a known feasible initialized model m
        model_name: Name of the model for the initialization
        human_read: Make the json file readable by a human
        wts: What to save, initially the values, but we might want something different. Check model_serializer tests for examples
    Returns:
        json_path: Path where json file is stored
    """

    dir_path = os.path.dirname(os.path.abspath(__file__))

    if starting_initialization:
        json_path = os.path.join(
            dir_path, model_name + '_initialization.json')
    else:
        if model_name != '':
            json_path = os.path.join(
                dir_path, model_name + '_initialization.json')
        else:
            json_path = os.path.join(
                dir_path, 'dsda_initialization.json')

    to_json(m, fname=json_path, human_read=human_read, wts=wts)

    return json_path

def build_master(num_ext,lower_b,upper_b,current,stage,D,use_random: bool=False):
    """
    Function that builds the master problem

    use_random: True if a random point will be generated for initializations when required. False if you want to use the deterministric strategy
        
    """
    initial={}
    randomp=[]
    if stage==1 or stage==2: #generate random number different from current value and points evaluated so far
        initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value
        # if fabs(float(len([el for el in D.keys() if all(el[n_e-1]>=lower_b[n_e] for   n_e in lower_b.keys()) and all(el[n_e-1]<=upper_b[n_e] for   n_e in lower_b.keys()) ]))-float(math.prod(upper_b[n_e]-lower_b[n_e]+1 for n_e in lower_b)))<=0.01: #if every point has been evaluated
        #     initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value
        # else:
        #     if use_random:
        #         #Generate random numbers
        #         while True:
        #             randomp=[random.randint(lower_b[n_e],upper_b[n_e]) for n_e in lower_b.keys()]  #This allows to consider difficult problems where, e.g., there is a big infeasible region with the same objective function values.
        #             if all([np.linalg.norm(np.array(randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
        #                 initial={n_e:randomp[n_e-1] for n_e in lower_b.keys()}
        #                 break
        #     else:
        #         #generate nonrandom numbers. It is better to go to the closest point that has not been evaluated (using e.g. a lexicographical ordering).  
        #         arrays=[range(lower_b[n_e],upper_b[n_e]+1) for n_e in lower_b.keys()]

        #         cart_prduct=list(product(*arrays)) #cartesian product, this also requires a lot of memory

        #         #TODO: after the cartesian product, I am organizing this with respect to the current value using a distance metric. Note that I can aslo explore points that are far away in the future.
        #         cart_prduct_sorted=cart_prduct#sorted(cart_prduct, key=lambda x: np.linalg.norm(np.array(list(x))-np.array(current)      )      ) #I am sorting to evaluate the closests point. I can also sort to evaluate the one that is far away (exploration!!!!!)
        #         for j in cart_prduct_sorted:
        #             non_randomp=list(j)
        #             if all([np.linalg.norm(np.array(non_randomp)-np.array(list(i)))>=0.1 for i in list(D.keys())]):
        #                 initial={n_e:non_randomp[n_e-1] for n_e in lower_b.keys()}
        #                 break
    else:
        initial={n_e:current[n_e-1] for n_e in lower_b.keys()} #use current value

    #print(initial)
    #Model
    m=pe.ConcreteModel(name='Master_problem')
    m.extset=pe.RangeSet(1,num_ext,1,doc='Set to organize external variables')
    #External variables
    def _boundsRule(m,extset):
        return (lower_b[extset],upper_b[extset])
    def _initialRule(m,extset):
        return initial[extset]    
    m.x=pe.Var(m.extset,within=pe.Integers,bounds=_boundsRule,initialize=_initialRule)

    # m.x1=pe.Var(within=pe.Integers, bounds=(lower_b[1],upper_b[1]),initialize=initial[1])
    # m.x2=pe.Var(within=pe.Integers, bounds=(lower_b[2],upper_b[2]),initialize=initial[2])

    #Known constraints (assumption!!! I know constraints a priori)
    #m.known=pe.Constraint(expr=m.x1-m.x2>=7)
    #m.known2=pe.Constraint(expr=m.x1>=9)
    #m.known3=pe.Constraint(expr=m.x2<=9)

    #Cuts
    m.cuts=pe.ConstraintList()

    #Objective function
    m.zobj=pe.Var()

    def obj_rule(m):
        return m.zobj 
    m.fobj=pe.Objective(rule=obj_rule,sense=pe.minimize)
    notevaluated=[round(k) for k in initial.values()]
    return m,notevaluated

def neighborhood_k_eq_2(dimension: int = 2) -> dict:
    """
    Function creates a k=2 neighborhood of the given dimension
    Args:
        dimension: Dimension of the neighborhood
    Returns:
        directions: Dictionary contaning in each item a list with a direction within the neighborhood
    """

    num_neigh = 2*dimension
    neighbors = np.concatenate(
        (np.eye(dimension, dtype=int), -np.eye(dimension, dtype=int)), axis=1)
    directions = {}
    for i in range(num_neigh):
        direct = []
        directions[i+1] = direct
        for j in range(dimension):
            direct.append(neighbors[j, i])
    return directions

def preprocess_problem(m, simple: bool = True):
    """
    Function that applies certain tranformations to the mdoel to first verify that it is not trivially 
    infeasible (via FBBT) and second, remove extra constraints to help NLP solvers
    Args:
        m: MI(N)LP model that is going to be preprocessed
        simple: Boolean variable to carry on a simple preprocessing (only FBBT) or a more complete one, prone to fail
    Returns:

    """
    if not simple:
        pe.TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        pe.TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        pe.TransformationFactory(
            'contrib.constraints_to_var_bounds').apply_to(m)
        pe.TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        pe.TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        pe.TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            m, tmp=False, ignore_infeasible=True)
    fbbt(m)