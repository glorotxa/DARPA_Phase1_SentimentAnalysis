#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <vector>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: run_all [options] training_set_file test_file output_file\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	0 -- L2-regularized logistic regression\n"
	"	1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	4 -- multi-class support vector classification by Crammer and Singer\n"
	"	5 -- L1-regularized L2-loss support vector classification\n"
	"	6 -- L1-regularized logistic regression\n"
	"-c cost : set the parameter C (default 1)\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 1, 3, and 4\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,\n"
	"		where f is the primal function (default 0.01)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"-l : size of the training subset (default 1000)\n"
	"-r : number of runs to perform (default 1)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
  int len;
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
    {
      max_line_len *= 2;
      line = (char *) realloc(line,max_line_len);
      len = (int) strlen(line);
      if(fgets(line+len,max_line_len-len,input) == NULL)	
	break;
    }
  return line;
}


////////////////

struct parameter param;
double bias;
int trnsz = 1000;
int nb_runs = 1;

struct problem* prob;
struct problem* tprob;

struct model* model_;
int flag_predict_probability=0;

FILE *output;

////////////////////

// read in a problem (in libsvm format)
struct problem *read_problem(const char *filename)
{
  struct feature_node *x_space;
  problem *prob= Malloc(problem,1);
  
  int max_index, inst_max_index, i;
  long int elements, j;
  FILE *fp = fopen(filename,"r");
  char *endptr;
  char *idx, *val, *label;
  
  if(fp == NULL)
    {
      fprintf(stderr,"can't open input file %s\n",filename);
      exit(1);
    }
  
  prob->l = 0;
  elements = 0;
  max_line_len = 1024;
  line = Malloc(char,max_line_len);
  while(readline(fp)!=NULL)
    {
      char *p = strtok(line," \t"); // label
      
      // features
      while(1)
	{
	  p = strtok(NULL," \t");
	  if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	    break;
	  elements++;
	      }
      elements++; // for bias term
      prob->l++;
    }
  rewind(fp);
  
  prob->bias=bias;
  
  prob->y = Malloc(int,prob->l);
  prob->x = Malloc(struct feature_node *,prob->l);
  x_space = Malloc(struct feature_node,elements+prob->l);
  
  max_index = 0;
  j=0;
  for(i=0;i<prob->l;i++)
    {
      inst_max_index = 0; // strtol gives 0 if wrong format
      readline(fp);
      prob->x[i] = &x_space[j];
      label = strtok(line," \t");
      prob->y[i] = (int) strtol(label,&endptr,10);
      if(endptr == label)
	exit_input_error(i+1);
      
      while(1)
	{
	  idx = strtok(NULL,":");
	  val = strtok(NULL," \t");

	  if(val == NULL)
	    break;
	  
	  errno = 0;
	  x_space[j].index = (int) strtol(idx,&endptr,10)+1;
	  if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
	    exit_input_error(i+1);
	  else
	    inst_max_index = x_space[j].index;
	  
	  errno = 0;
	  x_space[j].value = strtod(val,&endptr);
	  if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
	    exit_input_error(i+1);
	  
	  ++j;
	}
      
      if(inst_max_index > max_index)
	max_index = inst_max_index;
      
      if(prob->bias >= 0)
	x_space[j++].value = prob->bias;
      
      x_space[j++].index = -1;
    }
  
  if(prob->bias >= 0)
    {
      prob->n=max_index+1;
      for(i=1;i<prob->l;i++)
	(prob->x[i]-2)->index = prob->n; 
      x_space[j-2].index = prob->n;
    }
  else
    prob->n=max_index;
  
  fclose(fp);
  return prob;

}


// extract a subproblem
struct problem *extract_subprob(const struct problem *aprob, int start, int len)
{
  problem *sprob= Malloc(problem,1);

  sprob->l = len;
  sprob->n = aprob->n;
  sprob->bias=aprob->bias;
  
  sprob->y = Malloc(int,sprob->l);
  sprob->x = Malloc(struct feature_node *,sprob->l);
  
  for (int i=0; i<sprob->l; i++)
    {
      sprob->y[i]=aprob->y[start+i];
      sprob->x[i]=aprob->x[start+i];
    }

  return sprob;

}

void parse_command_line(int argc, char **argv)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	char input_file_name[1024];
	char test_file_name[1024];

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
		  exit_with_help();
		switch(argv[i-1][1])
		  {
		  case 'b':
		    flag_predict_probability = atoi(argv[i]);
		    break;

		  case 's':
		    param.solver_type = atoi(argv[i]);
		    break;
		    
		  case 'c':
		    param.C = atof(argv[i]);
		    break;
		    
		  case 'e':
		    param.eps = atof(argv[i]);
		    break;
		    
		  case 'B':
		    bias = atof(argv[i]);
		    break;

		  case 'w':
		    ++param.nr_weight;
		    param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
		    param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
		    param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
		    param.weight[param.nr_weight-1] = atof(argv[i]);
		    break;
		    
		  case 'q':
		    print_func = &print_null;
		    i--;
		    break;
		    
		  case 'r':
		    nb_runs = atoi(argv[i]);            
		    break;

		  case 'l':
		    trnsz = atoi(argv[i]);            
		    break;

		  default:
		    fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
		    exit_with_help();
		    break;
		  }
	}
	
	set_print_string_function(print_func);
	
	// determine filenames
	if(i+2>=argc)
		exit_with_help();
	
	printf("reading train file %s\n",argv[i]);
	strcpy(input_file_name, argv[i]);
	prob=read_problem(input_file_name);

	printf("reading test file %s\n",argv[i+1]);
	strcpy(test_file_name, argv[i+1]);
	tprob=read_problem(test_file_name);

	output = fopen(argv[i+2],"a");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if(param.eps == INF)
	  {
	    if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
	      param.eps = 0.01;
	    else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS)
	      param.eps = 0.1;
	    else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
	      param.eps = 0.01;
	  }
}

// return classification error and the normalized difference between predicted and true sentiment
std::pair<double, double> do_predict(const struct problem *test_prob, struct model* model_)
{
  double acc = 0;
  double clse=0;
  int total = 0;
  double *prob_estimates=NULL;
  int *labels=NULL;
  int nr_class=get_nr_class(model_);
  if(flag_predict_probability)
    {
      if(!check_probability_model(model_))
	{
	  fprintf(stderr, "probability output is only supported for logistic regression\n");
	  exit(1);
	}
      
      labels=(int *) malloc(nr_class*sizeof(int));
      get_labels(model_,labels);
      prob_estimates = (double *) malloc(nr_class*sizeof(double));
    }

  int l = test_prob->l;
  int i = 0;
  for(i=0; i<l; i++)
    {
      int predict_label = 0;
      int target_label=test_prob->y[i];
      feature_node *xi = test_prob->x[i];
      if(flag_predict_probability)
	{
	  int j;
	  predict_label = predict_probability(model_,xi,prob_estimates);
	  double predict_score=0;
	  for(j=0;j<model_->nr_class;j++)
	    predict_score+=prob_estimates[j]*labels[j];
	  //double acc_max= fabs(target_label-3)+2;
	  //acc+=(acc_max-sqrt((predict_score - target_label)*(predict_score - target_label)))/acc_max;
	  acc += (predict_score - target_label) * (predict_score - target_label);
	  if (predict_label!=target_label)
	    clse++;
	}
      else
	{
	  predict_label = predict(model_,xi);
	  //double acc_max= fabs(target_label-3)+2;
	  //acc+=(acc_max-sqrt((predict_label - target_label)*(predict_label - target_label)))/acc_max;
          acc += (predict_label - target_label) * (predict_label - target_label);
          if (predict_label!=target_label)
	    clse++;
	}
      ++total;
    }
  if(flag_predict_probability)
    {
      free(prob_estimates);
      free(labels);
    }
  //printf("Error = %g%% (%d/%d)\n",(double) (total-correct)/total*100,total-correct,total);
  return std::make_pair(clse/total,acc/total) ;
}





/////////////////


int main(int argc, char **argv)
{
  const char *error_msg;
  parse_command_line(argc, argv); // also load data
  error_msg = check_parameter(prob,&param);
  
  if(error_msg)
    {
      fprintf(stderr,"Error: %s\n",error_msg);
      exit(1);
    }
	

  std::vector< std::pair<double, double> > test_errors(nb_runs);
  std::vector< std::pair<double, double> > train_errors(nb_runs);
  double trn_mean=0;
  double tst_mean=0;
  double mse_trn_mean=0;
  double mse_tst_mean=0;
  int *start = NULL;

  // perform runs
  for (int run=0; run<nb_runs; run++)
    {

      if ((trnsz>=prob->l) || (trnsz<=0))
	{
	  fprintf(stderr,"\nRun %d (from 0 to %d)\n", run, prob->l-1);

	  //train
	  model_=train(prob, &param);
	  
	  // test
	  test_errors[run]=do_predict(tprob, model_);
	  train_errors[run]=do_predict(prob, model_);
	}
      else
	{
          // select all the splits before optimizing
          if(run == 0)
            {
              start = Malloc(int,nb_runs); 
              for (int run2=0; run2<nb_runs; run2++)
                start[run2] = (rand() % (prob->l-trnsz));
            }
	  // select examples
	  fprintf(stderr,"\nRun %d (from %d to %d)\n", run, start[run], start[run]+trnsz-1);
	  struct problem* subprob=extract_subprob(prob, start[run], trnsz);
	  
	  //train
	  model_=train(subprob, &param);
	  
	  // test
	  test_errors[run]=do_predict(tprob, model_);
	  train_errors[run]=do_predict(subprob, model_);
	  free(subprob->y);
	  free(subprob->x);
	}

      tst_mean+=test_errors[run].first;
      printf("Test  classification ERROR = %g\n",test_errors[run].first);
      trn_mean+=train_errors[run].first;
      printf("Train classification ERROR = %g\n",train_errors[run].first);

      mse_tst_mean+=test_errors[run].second;
      printf("Test  normalized ACCURACY (ET requirement) = %g\n",test_errors[run].second);
      mse_trn_mean+=train_errors[run].second;
      printf("Train normalized ACCURACY (ET requirement) = %g\n",train_errors[run].second);

      //destroy model
      free_and_destroy_model(&model_);
      destroy_param(&param);
      
    }
  double trn_var=0;
  double tst_var=0;
  tst_mean=tst_mean/nb_runs;
  trn_mean=trn_mean/nb_runs;

  double mse_trn_var=0;
  double mse_tst_var=0;
  mse_tst_mean=mse_tst_mean/nb_runs;
  mse_trn_mean=mse_trn_mean/nb_runs;

  for (int run=0; run<nb_runs; run++)
    {
      tst_var+=(test_errors[run].first-tst_mean)*(test_errors[run].first-tst_mean);
      trn_var+=(train_errors[run].first-trn_mean)*(train_errors[run].first-trn_mean);

      mse_tst_var+=(test_errors[run].second-mse_tst_mean)*(test_errors[run].second-mse_tst_mean);
      mse_trn_var+=(train_errors[run].second-mse_trn_mean)*(train_errors[run].second-mse_trn_mean); 
    }
  trn_var=sqrt(trn_var/nb_runs);
  tst_var=sqrt(tst_var/nb_runs);

  mse_trn_var=sqrt(mse_trn_var/nb_runs);
  mse_tst_var=sqrt(mse_tst_var/nb_runs);
  
  fprintf(stderr,"\nOVERALL TEST  ERROR on %d ex (%d runs): %g +/- %g\n", trnsz, nb_runs, tst_mean, tst_var);
  fprintf(stderr,"OVERALL TRAIN ERROR on %d ex (%d runs): %g +/- %g\n", trnsz, nb_runs, trn_mean, trn_var);

  fprintf(stderr,"\nOVERALL TEST  normalized ACCURACY (ET requirement) on %d ex (%d runs): %g +/- %g\n", trnsz, nb_runs, mse_tst_mean, mse_tst_var);
  fprintf(stderr,"OVERALL TRAIN normalized ACCURACY (ET requirement) on %d ex (%d runs): %g +/- %g\n", trnsz, nb_runs, mse_trn_mean, mse_trn_var);

  fprintf(output,"%d %g %g %g %g %g %g %g %g\n", trnsz, trn_mean, trn_var, tst_mean, tst_var,
	  mse_trn_mean, mse_trn_var, mse_tst_mean, mse_tst_var);

  //clean all
  free(prob->y);
  free(prob->x);
  
  free(tprob->y);
  free(tprob->x);
  
  free(line);
  fclose(output);
  
  return 0;
}


