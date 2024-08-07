#property copyright "Jaydin Gulley"

#include <Math\Stat\Normal.mqh>
#include <Math\Stat\Uniform.mqh>

// inspired by https://github.com/cknd/pyESN
enum ACTIVATION_FUNCTIONS{ Identity }; //,Sigmoid,Relu,Tanh,SoftMax
// Only Identity implemented
enum INVERSE_ACTIVATION_FUNCTIONS{ InverseIdentity };//,InverseSigmoid,InverseRelu,InverseTanh,InverseSoftMax

void print(matrix<double>&m, string name){

   Print(name);
   Print(m);
   Print(name);
   Print(m.Rows(),"<--- Row -:",name,":- Col --->",m.Cols());

}



matrix<double> hstack(matrix<double> &n, matrix<double> &m){


       matrix<double> m_1 = n;
       
       matrix<double> m_2 = m;
       
       
       matrix<double> res(n.Rows(), n.Cols() + m.Cols());
       
          
         for (unsigned int i = 0; i < n.Cols(); i++) {
         vector<double> temp_v = n.Col(i);
         res.Col(temp_v, i);
         }
         
         for (unsigned int i = 0; i < m.Cols(); i++) {
             vector<double> temp_v = m.Col(i);
             res.Col(temp_v, i + n.Cols());
         }
                

   return res;



}


matrix<double> vstack(matrix<double> &n, matrix<double> &m){


       matrix<double> m_1 = n;
       
       matrix<double> m_2 = m;
       
       
       matrix<double> res(n.Rows()+m.Rows(), n.Cols() );
       
          
         for (unsigned int i = 0; i < n.Rows(); i++) {
         vector<double> temp_v = n.Row(i);
         res.Row(temp_v, i);
         }
         
         for (unsigned int i = 0; i < m.Rows(); i++) {
             vector<double> temp_v = m.Row(i);
             res.Row(temp_v, i + n.Rows());
         }
                

   return res;



}



matrix<double> extended_states_Flatten( matrix<double> &extended_states, int transient ){

 matrix<double> resw(extended_states.Rows() - transient, extended_states.Cols());


for( unsigned int i = 0 + transient ; i< extended_states.Rows() ; i++){

   vector<double> temp_t_d = extended_states.Row(i);
   
   resw.Row(temp_t_d,i-transient);


}


 return resw;

}
matrix<double> teacher_scaled_Flatten( matrix<double> &teacher_scaled, int transient ){

 matrix<double> resw(teacher_scaled.Rows() - transient, teacher_scaled.Cols());


for( unsigned int i = 0 + transient ; i< teacher_scaled.Rows() ; i++){

   vector<double> temp_t_d = teacher_scaled.Row(i);
   
   resw.Row(temp_t_d,i-transient);


}


 return resw;

}

matrix<double> correct_dimension(matrix<double> &s, int targetlength){



      

      if( s.Rows() ==1 && s.Cols() ==1){
         matrix<double> newM(1,targetlength);
         newM.Fill(s.Mean());  
      
            return newM;
            
      }



      if( (s.Rows() ==1 && s.Cols() ==0) && (s.Rows() ==0 && s.Cols() ==1)){
      
      
         if( (s.Rows() != targetlength   && s.Cols() ==1)  || (s.Cols() != targetlength && s.Rows()==1)  ){


            //Print("arg must have length ",targetlength);         
         
         }
      
      
      
      }
      matrix<double> newM(s.Rows(),s.Cols());
      newM = s;
      return newM;
   

}


class ESN{


      private:
      
      void   MatrixRandom(matrix<double> & m);
      void   Matrixsparsity(matrix<double> & m);
      void   MatrixTanh(matrix<double> &m);      
      int    n_inputs;
      int    n_outputs;
      int    n_reservoir;
      double n_sparsity ;
      double n_spectral_radius;
      double n_noise ;
      matrix<double> input_shift_M ;
      matrix<double> input_scaling_M ;
      bool   n_teacher_forcing;
      bool   n_input_scaling_on;
      double n_teacher_scaling;
      double n_teacher_shift;
      bool   n_teacher_shift_on;
      bool   n_teacher_scaling_on;
      bool   n_input_shift_on;
      matrix<double> M_laststate;
      matrix<double> M_lastinput;
      matrix<double> M_lastoutput;
      matrix<double> M_W_out;
      bool m_verbose;
      
      
      ACTIVATION_FUNCTIONS n_out_activation;      
      INVERSE_ACTIVATION_FUNCTIONS n_inverse_out_activation;
      matrix<double> m_W;
      matrix<double> m_W_in;
      matrix<double> m_W_feedb;
      void initWeights();
      matrix<double> _scale_inputs(matrix<double> &inputs_M);
      matrix<double> _scale_teacher(matrix<double> &teacher_M);
      matrix<double> _unscale_teacher(matrix<double> &teacher_scaled_M);
      matrix<double> _update(matrix<double> &state, matrix<double> &input_pattern, matrix<double> &output_pattern );
      
      
      
      public:
      
      
      ESN(int inputs,int  outputs, int reservoir, double spectral_radius,double sparsity,double noise,matrix<double> &input_shift,bool input_shift_on,matrix<double> &input_scaling, bool input_scaling_on,bool teacher_forcing,double teacher_scaling,bool teacher_scaling_on,double teacher_shift,bool teacher_shift_on,  ACTIVATION_FUNCTIONS out_activation,INVERSE_ACTIVATION_FUNCTIONS inverse_out_activation, bool Verbose );
      
      matrix<double>Fit(matrix<double>&inputs , matrix<double>&outputs );

      matrix<double>predict(matrix<double>&inputs, bool continuation);
         


};

ESN::ESN(int inputs,int  outputs, int reservoir, double spectral_radius,double sparsity,double noise,matrix<double> &input_shift,bool input_shift_on,matrix<double> &input_scaling, bool input_scaling_on,bool teacher_forcing,double teacher_scaling,bool teacher_scaling_on,double teacher_shift,bool teacher_shift_on,  ACTIVATION_FUNCTIONS out_activation,INVERSE_ACTIVATION_FUNCTIONS inverse_out_activation, bool Verbose ){

  n_inputs = inputs;
  n_outputs = outputs;
  n_reservoir = reservoir;
  n_spectral_radius = spectral_radius;
  n_sparsity = sparsity;
  n_noise = noise;
  input_shift_M = correct_dimension(input_shift,n_inputs);
  n_input_shift_on= input_shift_on;
  input_scaling_M =correct_dimension(input_scaling,n_inputs);
  n_teacher_forcing = teacher_forcing;
  n_teacher_scaling = teacher_scaling;
  n_input_scaling_on = input_scaling_on;
  n_teacher_scaling_on =teacher_scaling_on;
  n_teacher_shift = teacher_shift;
  n_out_activation = out_activation;
  n_teacher_shift_on = teacher_shift_on; 
  n_inverse_out_activation = inverse_out_activation;
  m_verbose = Verbose;
  
  
  initWeights();
};


void ESN::initWeights(void){

   matrix<double> W(n_reservoir,n_reservoir);
   MatrixRandom(W);
   W =W -.5;
   Matrixsparsity(W);
   matrix<double> tempW= (W.Transpose()+ W)/2;
   
   double radius=tempW.Norm(MATRIX_NORM_SPECTRAL);
   
   Print(radius,"<----radius");
   
   matrix<double> spec(n_reservoir,n_reservoir);
   
   spec = spec.Fill(n_spectral_radius/radius);
   
   W = W*spec;
   
   m_W = W;
   
    
   
   matrix<double> W_in_t(n_reservoir,n_inputs);
   MatrixRandom(W_in_t);
   
   
   
   m_W_in = W_in_t*2-1;
   
   
 
   
   matrix<double> W_feedb_t(n_reservoir,n_outputs);
   MatrixRandom(W_feedb_t);
   m_W_feedb = W_feedb_t*2-1;
  
   
   
   
   
};

matrix<double>ESN::_update(matrix<double> &state,matrix<double> &input_pattern,matrix<double> &output_pattern){


   matrix<double> preactivation;

   if( n_teacher_forcing == true){
   
     
      
         
       preactivation = (m_W.MatMul( state) + m_W_in.MatMul(input_pattern)+ m_W_feedb.MatMul( output_pattern));
     
       
   
   }else{
   
      preactivation = m_W.MatMul( state) + m_W_in.MatMul(input_pattern);
  
   }











   

   MatrixTanh(preactivation);
    
   
   
   
   matrix<double> tempRand(n_reservoir,1);
   
   
   MatrixRandom(tempRand);
   
   matrix<double> tempRand_noise(n_reservoir,1);
   
   tempRand_noise = tempRand_noise.Fill(n_noise);
  
    matrix<double> tempRand_min(n_reservoir,1);
   
   tempRand_noise = tempRand_min.Fill(0.5);
   
   matrix<double> res = preactivation+tempRand_noise*(tempRand -tempRand_min);
   
   
   return   res;

};




matrix<double> ESN::_scale_inputs(matrix<double> &inputs_M){

   matrix<double> inputs_t;

if( n_input_scaling_on == true  ){


      vector<double> input_scaling_V =input_scaling_M.Diag();
      
      matrix<double> input_scaling_Diag_M;
      input_scaling_M.Swap(input_scaling_V );
      matrix<double> input_scaling_M_T = input_scaling_M.Transpose();
      
      inputs_t= inputs_M.MatMul( input_scaling_M_T);
      return inputs_t.Transpose();

}

if( n_input_shift_on== true   ){


      

      inputs_t= inputs_M + input_shift_M;
      return inputs_t;

}
 inputs_t = inputs_M;
return inputs_M;

}



matrix<double>ESN::_scale_teacher(matrix<double> &teacher_M){

   matrix<double> teacher_t;

if(  n_teacher_scaling_on == true ){


      

      teacher_t= teacher_M*input_scaling_M;
      return teacher_t ;
}
if(  n_teacher_shift_on == true ){


      

      teacher_t= teacher_M + input_shift_M;
      return teacher_t ;
}

return teacher_M ;

}






matrix<double>ESN::_unscale_teacher(matrix<double>&teacher_scaled_M){



   matrix<double> teacher_t;


if(  n_teacher_shift_on  == true){


      

      teacher_t= teacher_scaled_M - n_teacher_shift;
      return teacher_t;
}





if(  n_teacher_scaling_on  == true ){


      

      teacher_t= teacher_scaled_M/n_teacher_scaling;
      return teacher_t;
}


return teacher_scaled_M;



}








matrix<double> ESN::Fit( matrix<double>&inputs,matrix<double>&outputs){





         
         matrix<double> inputs_scaled_t = _scale_inputs(inputs);
         matrix<double> teacher_scaled_t = _scale_teacher(outputs);
        
       
         matrix<double> states(inputs.Rows(),n_reservoir); //50x500
         states = states.Fill(0);
         
         
         
         
         
         for( int n=1 ; n<inputs.Rows(); n++){
              
              
                
            
            
         
            
            vector<double> states_t_row = states.Row(n);
            vector<double> inputs_scaled_t_row = inputs_scaled_t.Row(n-1);
            vector<double> teacher_scaled_t_row = teacher_scaled_t.Row(n-1);
            
            
            
            matrix<double> states_t_row_m ;
            matrix<double> inputs_scaled_t_row_m ;
            matrix<double> teacher_scaled_t_row_m ;
            
            
            states_t_row.Swap(states_t_row_m);
            inputs_scaled_t_row.Swap(inputs_scaled_t_row_m);
            teacher_scaled_t_row.Swap(teacher_scaled_t_row_m);
            
            matrix<double> states_t_row_m_T = states_t_row_m.Transpose(); 
            matrix<double> inputs_scaled_t_row_m_T =inputs_scaled_t_row_m.Transpose();
            matrix<double> teacher_scaled_t_row_m_T = teacher_scaled_t_row_m.Transpose();
            
            matrix<double> update_row = _update(states_t_row_m_T,inputs_scaled_t_row_m_T,teacher_scaled_t_row_m_T);
            
            vector<double> update_row_v;
            update_row_v.Swap(update_row);
            states.Row(update_row_v,n);
            
         }
         
         
         int transient =0;
         
         if(int( double(inputs.Cols())/10) < 100 ){transient = int(double(inputs.Cols())/10); }
         if(int( double(inputs.Cols())/10) >= 100 ){transient = 100; }
    
         
         
         
         matrix<double> extended_states = hstack(states,inputs_scaled_t);
       
         
         
         
         matrix<double> extended_states_flat = extended_states_Flatten(extended_states,transient);
         matrix<double> teacher_scaled_t_flat = teacher_scaled_Flatten(teacher_scaled_t,transient);
      
         matrix<double> extended_states_flat_pinv = extended_states_flat.PInv();
         
         if( n_inverse_out_activation == InverseIdentity){
         
              ///pass
         
         }
         else{
         
         //Print("no dev");
         }

            matrix<double> W_out_base = extended_states_flat_pinv.MatMul(teacher_scaled_t_flat);
            W_out_base = W_out_base.Transpose();
            
            M_W_out = W_out_base;
            
         
            vector<double> temp_states_v = states.Row(states.Rows()-1);
           
            M_laststate.Swap(temp_states_v);
            M_laststate =M_laststate.Transpose();
            
            vector<double> temp_input_v = inputs.Row(inputs.Rows()-1);
            M_lastinput.Swap(temp_input_v); 
            M_lastinput =M_lastinput.Transpose();
            
            
            vector<double> temp_output_v = teacher_scaled_t.Row(teacher_scaled_t.Rows()-1);
            M_lastoutput.Swap(temp_output_v); 
            M_lastoutput = M_lastoutput.Transpose();
            
            
            matrix<double> level_1 = extended_states.MatMul(W_out_base.Transpose());
            
           if( n_out_activation == Identity){
         
              ///pass
         
         }
         else{
         
         //Print("no dev");
         }
         
         matrix<double> pred_train = _unscale_teacher(level_1);
         
         
         matrix<double> delta = (pred_train - outputs);
         matrix<double> delta_squared = delta*delta;
         
         double Error = sqrt(delta_squared.Mean());
         
         Print(Error,"<<<<----- MSE");
         
         
         return pred_train;
};

matrix<double> ESN::predict(matrix<double> &inputs, bool continuation){

      
      matrix<double> res; 
      
      ulong n_samples = inputs.Rows();
      
      matrix<double> laststate ;
      matrix<double> lastinput;
      matrix<double> lastoutput ;
      if( continuation == true){
      
            laststate = M_laststate;
            lastinput = M_lastinput;
            lastoutput = M_lastoutput;
           
             
      }else{
      
            laststate.Resize(n_reservoir,1);
            laststate=laststate.Fill(0);
            lastinput.Resize(n_inputs,1) ;
            lastinput=lastinput.Fill(0);
            lastoutput.Resize(n_outputs,1);
            lastoutput=lastoutput.Fill(0);
      
      }
      
      
      matrix<double> scaled_inputs_M = _scale_inputs(inputs);
      matrix<double> inputs_p = vstack(lastinput.Transpose(),scaled_inputs_M.Transpose());  
     
      
      
      matrix<double> temp_states(n_samples,n_reservoir);
      temp_states=temp_states.Fill(0);
      matrix<double> states_p = vstack( laststate.Transpose(),temp_states);
      
     
      
      matrix<double> temp_outputs(n_samples,n_outputs);
      temp_outputs=temp_outputs.Fill(0);
      matrix<double> outputs_p = vstack(lastoutput.Transpose(), temp_outputs.Transpose());

       for( int n=0 ; n<n_samples; n++){
         
            
            
            
            vector<double> states_p_row = states_p.Row(n);
            vector<double> inputs_p_row = inputs_p.Row(n+1);
            vector<double> outputs_p_row = outputs_p.Row(n);
            
            
            
            matrix<double> states_p_M ;
            matrix<double> inputs_p_M ;
            matrix<double> outputs_p_M ;
            
            
            states_p_M.Swap(states_p_row);
            inputs_p_M.Swap(inputs_p_row);
            outputs_p_M.Swap(outputs_p_row);
            
            matrix<double> states_p_M_T=states_p_M.Transpose();
            matrix<double> inputs_p_M_T=inputs_p_M.Transpose();
            matrix<double> outputs_p_M_T=outputs_p_M.Transpose();
            
            
            matrix<double> update_row = _update(states_p_M_T,inputs_p_M_T,outputs_p_M_T);
            
            vector<double> update_row_v;
            update_row_v.Swap(update_row);
            states_p.Row(update_row_v,n+1);
         
         
         
         
         
            vector<double> states_p_v = states_p.Row(n+1);
            vector<double> inputs_p_v = inputs_p.Row(n+1);
            
            
            matrix<double> states_p_M_temp;
            matrix<double> inputs_p_M_temp;
            
           
            states_p_M_temp.Swap(states_p_v);
            inputs_p_M_temp.Swap(inputs_p_v);
            
            states_p_M_temp =states_p_M_temp.Transpose();  
            inputs_p_M_temp = inputs_p_M_temp.Transpose();  
             
            
            matrix<double> temp_concat = vstack(states_p_M_temp,inputs_p_M_temp);
           
            matrix<double> temp_W_out = M_W_out.MatMul(temp_concat);
            
            temp_W_out = temp_W_out.Transpose();
            
             if( n_out_activation == Identity){
               
                    ///pass
               
               }
               else{
               
               //Print("no dev");
               }
               
               
               vector<double> temp_W_out_v;
               temp_W_out_v.Swap(temp_W_out);
               
               
               outputs_p.Row(temp_W_out_v,n+1);
               
                
                  
      }
            
            matrix<double> flatten_outputs = extended_states_Flatten(outputs_p,1);
             
             if( n_out_activation == Identity){
            
                 ///pass
            
            }
            else{
            
            //Print("no dev");
            }
            
            
            
            res= _unscale_teacher(flatten_outputs);
            
            
            
        return res;   
};



void ESN::MatrixTanh(matrix<double> &m)
{
    for(ulong r = 0; r < m.Rows(); r++)
    {
        for(ulong c = 0; c < m.Cols(); c++)
        {
            m[r][c] = MathTanh(m[r][c]); // Use MathTanh if available
        }
    }
}
void ESN::MatrixRandom(matrix<double> &m)
{
    int error;
    for(ulong r = 0; r < m.Rows(); r++)
    {
        for(ulong c = 0; c < m.Cols(); c++)
        {
            m[r][c] = MathRandomUniform(0, 1, error);
            if (error != 0) {
                Print("Random error at (", r, ",", c, ")");
            }
        }
    }
}
void  ESN::Matrixsparsity(matrix<double> &m)
 {
   int error;
  for(ulong r=0; r<m.Rows(); r++)
   {
    for(ulong c=0; c<m.Cols(); c++)
     {
      
      if( MathRandomUniform(0,1,error) < n_sparsity){
         
         m[r][c]= 0;
      
      }
      
      
      
       
     }
   }
 }
