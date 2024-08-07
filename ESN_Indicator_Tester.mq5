//+------------------------------------------------------------------+
//|                                         ESN_Indicator_Tester.mq5 |
//|                                                    Jaydin Gulley |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+



#include <ESN.mqh>


input group "---General Settings---";
input double Lot = 1.05;
double pip = _Point * 10;
input int TPinPips = 5000;
input int SLinPips = 5000;
input int MagicNumberBUY = 66691;
input int MagicNumberSELL = 66692;
input int Verbose_         = false;


input group "---ESN Settings---";
input int Size_of_Esn = 1;
input int Neurons_ = 500;
input int spartsity_ = 0;
input double noise_ = .001;
input double spectral_radius_ = 2.8;

input int HurstAndKappaDepth = 10;

double NumberOfLosses = 0;

int min_tracker = -1;

double Lot_Martingale = 0;

matrix<double> shift_(Size_of_Esn,Size_of_Esn);
matrix<double> shift_b = shift_.Fill(0.0);
matrix<double> scaling_(Size_of_Esn,Size_of_Esn);
matrix<double> scaling_b = scaling_.Fill(1.0);
bool teacher_scaling_on_b = false;
bool input_scaling_on_b = false;
bool teacher_forcing_on_b = true;
bool teacher_shift_on_b = false;
bool input_shift_on_b = false;
double scaler_C = 1;
double shift_C = 0;

double iPrice(string symbol, ENUM_TIMEFRAMES TF, int shifts){

   return (iClose(symbol,TF,shifts) + iOpen(symbol,TF,shifts))/2;

}

double Vol(int shifts,int depth_, ENUM_TIMEFRAMES TF){

   double sum = 0;
   
   for(int i = 0; i < depth_; i++){
   
      sum = sum + MathLog(iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i))*MathLog(iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i));
   
   }
   
   sum = (1/double(depth_))*sum;
   
   return MathSqrt(sum);

}

double Mu(int shifts, int depth_, ENUM_TIMEFRAMES TF){

   double sum = 0;
   
   for(int i = 0; i < depth_; i++){
   
      sum = sum+MathLog(iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i));
   
   }
   
   sum = (1/double(depth_))*sum;
   
   return sum;

}


double Scale_Vol(double val){


   return MathExp(MathExp(val))-1;

}

double unScale_Vol(double val){

   return MathLog(MathLog(val+1));

}

double Scale_Mu(double val){


   return MathExp(MathExp(val))-1;

}

double unScale_Mu(double val){

   return MathLog(MathLog(val+1));

}

double Z_Sum_t(int n, int t, ENUM_TIMEFRAMES TF){


   double sum = 0;
   
   for(int i = 0; i < t; i++){
   
      sum = sum + iPrice(_Symbol,TF,i) - Sum_m(n,TF);
   
   }
   
   return sum/double(n);
}


double std(int n, int t, ENUM_TIMEFRAMES TF){

   double sum = 0;
   
   for(int i = 0; i < t; i++){
   
      sum = sum + pow(iPrice(_Symbol,TF,i) - Sum_m(n,TF), 2);
   
   }
   
   return MathSqrt(sum/double(n));


}


double RS_n(int n, ENUM_TIMEFRAMES TF){

   double R_max = 0;
   double R_min = 0;
   for(int t = 1; t <= n; t++){
   
      double Zt = Z_Sum_t(n,t,TF);
      if(t == 0){
      
         R_max = Zt;
         R_min = Zt;
      
      } else {
      
         if(R_max < Zt){
            R_max = Zt;
         
         }
         if(R_min > Zt){
            R_min = Zt;
         }
      
      }
   
   }
   
   double Rt = R_max - R_min;
   double St = std(n,n,TF);
   
   return Rt/St;


}


double Hurst(int depth_power_of_2, ENUM_TIMEFRAMES TF){


   double top = MathLog(depth_power_of_2);
   double bottom = MathLog(2);
   
   int NumberOfpowersOf2 = int(top/bottom);
   
   matrix Var(NumberOfpowersOf2-2,2);
   vector EofRtdivSt(NumberOfpowersOf2-2);
   
   for(int i = 2; i < NumberOfpowersOf2; i++){
   
   
      int n = int(pow(2,i));
      
      double ln_n = MathLog(double(n));
      double ln_c = 1;
      double RS = RS_n(n,TF);
      
      Var[i-2][0] = ln_n;
      Var[i-2][1] = ln_c;
      EofRtdivSt[i-2] = RS;
   
   }
   
   vector res = Var.LstSq(EofRtdivSt);
   double H = res[1];
   return H;
   

}

double Sum_m(int n, ENUM_TIMEFRAMES TF){

   double sum = 0;
   for(int i = 0; i < n; i++){
   
      sum = sum + iPrice(_Symbol,TF,i);
   
   }
   
   return sum/double(n);

}

double Sum_n(int n, ENUM_TIMEFRAMES TF){

   double sum = 0;
   for(int i = 0; i < n; i++){
   
      sum = sum + MathAbs(iPrice(_Symbol,TF,i)-iPrice(_Symbol,TF,1));
   
   }
   
   return sum/double(n);

}

double E_Sum_n(int n, ENUM_TIMEFRAMES TF){


   double sum = 0;
   for(int i =0; i < n; i++){
   
      sum = sum + Sum_n(i,TF);
   
   }


   return sum/double(n);
}


double MeanAbsoluteDeviation(int n, ENUM_TIMEFRAMES TF){

   double sum = 0;
   for(int i =1; i < n; i++){
   
      sum = sum + MathAbs(Sum_n(i,TF) - E_Sum_n(i,TF));
   
   }


   return sum/double(n);
   

}


double Hvalues[4][5];  // Initialize with 0
double dxValues[4][300];  // Initialize with 0

ESN *ESNlist[4];


int OnInit()
  {
//---


      for(int i = 0; i < 4; i++){
      
         ESNlist[i] = new ESN(Size_of_Esn,Size_of_Esn,Neurons_,spectral_radius_,spartsity_,noise_,shift_b,input_shift_on_b,scaling_b,input_scaling_on_b,teacher_forcing_on_b,scaler_C,teacher_scaling_on_b,shift_C,teacher_shift_on_b,Identity,InverseIdentity,Verbose_);
      
      }
      
      for(int j = 0; j < 4; j++){
      
         for (int i = 0; i < 5; i++){
         
            Hvalues[j][i] = 1;
         
         }
         
         for(int i = 0; i < 300; i++){
         
            dxValues[j][i] = 1;
         
         }
      
      }
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

      for(int i = 0; i < 4; i++){
      
         delete ESNlist[i];
      
      }
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    datetime tm = TimeCurrent();
    MqlDateTime stm;
    TimeToStruct(tm, stm);

    if (stm.min != min_tracker && stm.min % 15 == 0)
    {
        min_tracker = stm.min;

        Print("|--------------------Acc--------------------|");

        int power_of_2 = int(pow(2, HurstAndKappaDepth));
        double H_future = Hurst(power_of_2, PERIOD_M15);

        Print("H_future: ", H_future);

        int RSI_handle = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
        double RSI_1_Array[];
        ArraySetAsSeries(RSI_1_Array, true);
        CopyBuffer(RSI_handle, 0, 0, 3, RSI_1_Array);

        Print("RSI_1_Array: ", RSI_1_Array[0], ", ", RSI_1_Array[1], ", ", RSI_1_Array[2]);

        int STO_handle = iStochastic(_Symbol, PERIOD_M15, 14, 3, 3, MODE_EMA, STO_LOWHIGH);
        double STO_1_Array[];
        ArraySetAsSeries(STO_1_Array, true);
        CopyBuffer(STO_handle, 0, 0, 3, STO_1_Array);

        Print("STO_1_Array: ", STO_1_Array[0], ", ", STO_1_Array[1], ", ", STO_1_Array[2]);

        int MACD_handle = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
        double MACD_1_Array[];
        ArraySetAsSeries(MACD_1_Array, true);
        CopyBuffer(MACD_handle, 0, 0, 3, MACD_1_Array);

        Print("MACD_1_Array: ", MACD_1_Array[0], ", ", MACD_1_Array[1], ", ", MACD_1_Array[2]);

        string Names[4] = {"Hurst", "RSI", "Stochastic Oscillator", "MACD"};
        double Vals[4] = {H_future + 1, RSI_1_Array[0] / 100 + 1, STO_1_Array[0] / 100 + 1, MathLog(MathLog(MACD_1_Array[0] + 500))};

        for (int value = 0; value < 4; value++)
{
    double temp[5];
    H_future = Vals[value];

    for (int i = 0; i < 5; i++)
    {
        temp[i] = Hvalues[value][i];
    }
    for (int i = 0; i < 5 - 1; i++)
    {
        Hvalues[value][i + 1] = temp[i];
    }
    Hvalues[value][0] = H_future;

    double H_curr = Hvalues[value][1];
    double H_prev = Hvalues[value][2];

    Print("H_curr: ", H_curr);
    Print("H_prev: ", H_prev);

    matrix<double> future_H_(Size_of_Esn, Size_of_Esn);
    matrix<double> future = future_H_.Fill(H_future);
    matrix<double> curr_H_(Size_of_Esn, Size_of_Esn);
    matrix<double> curr = curr_H_.Fill(H_curr);
    matrix<double> prev_H_(Size_of_Esn, Size_of_Esn);
    matrix<double> prev = prev_H_.Fill(H_prev);

    ESNlist[value].Fit(prev, curr);
    matrix HurstPred_M = ESNlist[value].predict(curr, true);

    double HurstPred;
    double correct_;
    double dx;

    if (value == 3)
    {
        HurstPred = HurstPred_M.Mean() - 500;  // Simplified transformation
        correct_ = future.Mean() - 500;  // Simplified transformation
        dx = MathAbs((MACD_1_Array[0] - 500) - HurstPred) / 100;  // Adjusting for transformation
    }
    else
    {
        HurstPred = HurstPred_M.Mean() -1;
        correct_ = future.Mean() - 1;
        dx = MathAbs(correct_ - HurstPred);
    }
    Print("dx: ", dx);

    double tempdx[300];

    for (int i = 0; i < 300; i++)
    {
        tempdx[i] = dxValues[value][i];
    }
    for (int i = 0; i < 299; i++)
    {
        dxValues[value][i + 1] = tempdx[i];
    }
    dxValues[value][0] = dx;

    double s = 0;
    for (int i = 0; i < 300; i++)
    {
        s += dxValues[value][i];
    }

    double average_dx = s / 300;
    double correct_percentage = 100 - (average_dx * 100);

    Print(correct_percentage, "% correct ", Names[value], " on average");
    
    // Trading logic based on prediction and dx
            if (dx < 1 && correct_percentage > 1 && (stm.hour < 16 && stm.hour >= 8) && (stm.day_of_week >= 1 && stm.day_of_week <= 4) && PositionsTotal() == 0)
            {
               Print("Hurst Pred: ", HurstPred, "Correct_: ", correct_);
                // Make a trade decision
                if (HurstPred > correct_)
                {
            double TakeProfit = pip * TPinPips;
            double StopLoss = pip * SLinPips;

            MqlTradeRequest myrequest;
            MqlTradeResult myresult;
            ZeroMemory(myrequest);
            ZeroMemory(myresult);

            myrequest.type = ORDER_TYPE_BUY;
            myrequest.action = TRADE_ACTION_DEAL;
            myrequest.sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) - StopLoss;
            myrequest.tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + TakeProfit;
            myrequest.symbol = _Symbol;
            myrequest.volume = Lot;
            myrequest.type_filling = ORDER_FILLING_FOK;
            myrequest.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            myrequest.magic = MagicNumberBUY;

            if (!OrderSend(myrequest, myresult)) {
                Print("Error in OrderSend (BUY): ", GetLastError());
            }
                }
                else
                {

            double TakeProfit = pip * TPinPips;
            double StopLoss = pip * SLinPips;

            MqlTradeRequest myrequesta;
            MqlTradeResult myresulta;
            ZeroMemory(myrequesta);
            ZeroMemory(myresulta);

            myrequesta.type = ORDER_TYPE_SELL;
            myrequesta.action = TRADE_ACTION_DEAL;
            myrequesta.sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + StopLoss;
            myrequesta.tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) - TakeProfit;
            myrequesta.symbol = _Symbol;
            myrequesta.volume = Lot;
            myrequesta.type_filling = ORDER_FILLING_FOK;
            myrequesta.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            myrequesta.magic = MagicNumberSELL;

            if (!OrderSend(myrequesta, myresulta)) {
                Print("Error in OrderSend (SELL): ", GetLastError());
            }
                }
            }


    }
}
    
}



//+------------------------------------------------------------------+
