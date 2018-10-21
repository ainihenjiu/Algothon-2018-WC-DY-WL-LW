//+------------------------------------------------------------------+
//|                                                         ARRA.mq4 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
      /*ARRA ENTRY CONDITION:
      If Current Volatility (ATR(20)) is greater than Volatility (ATR(20)) 10 hours ago:
      Enter a long trade when SMA(10) crosses SMA(40) from the bottom
      Enter a short trade when SMA(10) crosses SMA(40) from the top
   
      ARRA EXIT CONDITION:
      Exit the long trade when SMA(10) crosses SMA(40) from the top
      Exit the short trade when SMA(10) crosses SMA(40) from the bottom
      30 pips hard stop (30pips from initial entry price)
      Trailing stop of 30 pips
   
      ARRA POSITION SIZING:
      Based on account size
   */

#define SIGNAL_NONE 0
#define SIGNAL_BUY   1
#define SIGNAL_SELL  2
#define SIGNAL_CLOSEBUY 3
#define SIGNAL_CLOSESELL 4

#property copyright "Expert Advisor Builder"
#property link      "http://sufx.core.t3-ism.net/ExpertAdvisorBuilder/"

extern int MagicNumber = 12345;
extern bool SignalMail = False;
extern double Lots = 1.0;
extern int Slippage = 3;
extern bool UseStopLoss = True;
extern int StopLoss = 30;
extern bool UseTakeProfit = False;
extern int TakeProfit = 0;
extern bool UseTrailingStop = True;
extern int TrailingStop = 30;

extern int sma_short = 10;
extern int sma_long = 40;
extern int atr_period = 20;
extern int atr_shift = 11;

//Risk Variable and Trigger

extern bool isSizingOn = true;
extern double Risk = 1;

int P = 1;
int Order = SIGNAL_NONE;
int Total, Ticket, Ticket2;
double StopLossLevel, TakeProfitLevel, StopLevel;
double sma10_1, sma10_2, sma40_1, sma40_2;
double atr_current, atr_past;


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int init() {
   
   if(Digits == 5 || Digits == 3 || Digits == 1)P = 10;else P = 1; // To account for 5 digit brokers

   return(0);
   }
//+------------------------------------------------------------------+
//| Expert initialization function - END                             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit() {
   return(0);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function - END                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert start function                                            |
//+------------------------------------------------------------------+
int start() {

   Total = OrdersTotal();
   Order = SIGNAL_NONE;

   //+------------------------------------------------------------------+
   //| Variable Setup                                                   |
   //+------------------------------------------------------------------+

   sma10_1 = iMA(NULL, 0, sma_short, 0, MODE_SMA, PRICE_CLOSE, 1); // c
   sma10_2 = iMA(NULL, 0, sma_short, 0, MODE_SMA, PRICE_CLOSE, 2); // b
   sma40_1 = iMA(NULL, 0, sma_long, 0, MODE_SMA, PRICE_CLOSE, 1); // d
   sma40_2 = iMA(NULL, 0, sma_long, 0, MODE_SMA, PRICE_CLOSE, 2); // a
   
   atr_current = iATR(NULL, 0, atr_period, 1);    // ATR(20) now
   atr_past = iATR(NULL, 0, atr_period, atr_shift);      // ATR(20) 10 periods ago
   
   StopLevel = (MarketInfo(Symbol(), MODE_STOPLEVEL) + MarketInfo(Symbol(), MODE_SPREAD)) / P;  // Defining minimum StopLevel

   if (StopLoss < StopLevel) StopLoss = StopLevel;
   if (TakeProfit < StopLevel) TakeProfit = StopLevel;
   
   // Sizing 
   if (isSizingOn == true) {
      Lots = Risk * 0.01 * AccountBalance() / (MarketInfo(Symbol(),MODE_LOTSIZE) * StopLoss * P * Point); // Sizing Algo based on account size
     
   }
   //+------------------------------------------------------------------+
   //| Variable Setup - END                                             |
   //+------------------------------------------------------------------+

   //Checking position
   bool IsTrade = False;

   for (int i = 0; i < Total; i ++) {
      Ticket2 = OrderSelect(i, SELECT_BY_POS, MODE_TRADES);
      if(OrderType() <= OP_SELL &&  OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber) {
         IsTrade = True;
         if(OrderType() == OP_BUY) {
            //Close

            //+------------------------------------------------------------------+
            //| Signal Begin(Exit Buy)                                           |
            //+------------------------------------------------------------------+

            /* ARRA EXIT:
               Exit the long trade when SMA(10) crosses SMA(40) from the top
               Exit the short trade when SMA(10) crosses SMA(40) from the bottom
               Trailing stop of 30 pips: Hard stop 30 pips (from initial entry price)
               
            */
            
            if(sma10_2 > sma40_2 && sma40_1 >= sma10_1) Order = SIGNAL_CLOSEBUY; // Rule to EXIT a Long trade

            //+------------------------------------------------------------------+
            //| Signal End(Exit Buy)                                             |
            //+------------------------------------------------------------------+

            if (Order == SIGNAL_CLOSEBUY) {
               Ticket2 = OrderClose(OrderTicket(), OrderLots(), Bid, Slippage, MediumSeaGreen);
               if (SignalMail) SendMail("[Signal Alert]", "[" + Symbol() + "] " + DoubleToStr(Bid, Digits) + " Close Buy");
               IsTrade = False;
               continue;
            }
            //Trailing stop
            if(UseTrailingStop && TrailingStop > 0) {                 
               if(Bid - OrderOpenPrice() > P * Point * TrailingStop) {
                  if(OrderStopLoss() < Bid - P * Point * TrailingStop) {
                     Ticket2 = OrderModify(OrderTicket(), OrderOpenPrice(), Bid - P * Point * TrailingStop, OrderTakeProfit(), 0, MediumSeaGreen);
                     continue;
                  }
               }
            }
         } else {
            //Close

            //+------------------------------------------------------------------+
            //| Signal Begin(Exit Sell)                                          |
            //+------------------------------------------------------------------+

            if (sma40_2 > sma10_2 && sma10_1 >= sma40_1) Order = SIGNAL_CLOSESELL; // Rule to EXIT a Short trade

            //+------------------------------------------------------------------+
            //| Signal End(Exit Sell)                                            |
            //+------------------------------------------------------------------+

            if (Order == SIGNAL_CLOSESELL) {
               Ticket2 = OrderClose(OrderTicket(), OrderLots(), Ask, Slippage, DarkOrange);
               if (SignalMail) SendMail("[Signal Alert]", "[" + Symbol() + "] " + DoubleToStr(Ask, Digits) + " Close Sell");
               IsTrade = False;
               continue;
            }
            //Trailing stop
            if(UseTrailingStop && TrailingStop > 0) {                 
               if((OrderOpenPrice() - Ask) > (P * Point * TrailingStop)) {
                  if((OrderStopLoss() > (Ask + P * Point * TrailingStop)) || (OrderStopLoss() == 0)) {
                     Ticket2 = OrderModify(OrderTicket(), OrderOpenPrice(), Ask + P * Point * TrailingStop, OrderTakeProfit(), 0, DarkOrange);
                     continue;
                  }
               }
            }
         }
      }
   }

   //+------------------------------------------------------------------+
   //| Signal Begin(Entries)                                            |
   //+------------------------------------------------------------------+

   /* ARRA ENTRY:
      If Current Volatility (ATR(20)) is greater than Volatility (ATR(20)) 10 hours ago:
      Enter a long trade when SMA(10) crosses SMA(40) from the bottom
      Enter a short trade when SMA(10) crosses SMA(40) from the top
   */
   
   if (New_Bar()) if (atr_current > atr_past) {
   
      if (sma40_2 > sma10_2 && sma10_1 >= sma40_1) Order = SIGNAL_BUY; // Rule to ENTER a Long trade
   
      if (sma10_2 > sma40_2 && sma40_1 >= sma10_1) Order = SIGNAL_SELL; // Rule to ENTER a Short trade

   }
   
   //+------------------------------------------------------------------+
   //| Signal End                                                       |
   //+------------------------------------------------------------------+

   //Buy
   if (Order == SIGNAL_BUY) {
      if(!IsTrade) {
         //Check free margin
         if (AccountFreeMargin() < (1000 * Lots)) {
            Print("We have no money. Free Margin = ", AccountFreeMargin());
            return(0);
         }

         if (UseStopLoss) StopLossLevel = Ask - StopLoss * Point * P; else StopLossLevel = 0.0;
         if (UseTakeProfit) TakeProfitLevel = Ask + TakeProfit * Point * P; else TakeProfitLevel = 0.0;

         Ticket = OrderSend(Symbol(), OP_BUY, Lots, Ask, Slippage, StopLossLevel, TakeProfitLevel, "Buy(#" + MagicNumber + ")", MagicNumber, 0, DodgerBlue);
         if(Ticket > 0) {
            if (OrderSelect(Ticket, SELECT_BY_TICKET, MODE_TRADES)) {
				Print("BUY order opened : ", OrderOpenPrice());
                if (SignalMail) SendMail("[Signal Alert]", "[" + Symbol() + "] " + DoubleToStr(Ask, Digits) + " Open Buy");
			} else {
				Print("Error opening BUY order : ", GetLastError());
			}
         }
         return(0);
      }
   }

   //Sell
   if (Order == SIGNAL_SELL) {
      if(!IsTrade) {
         //Check free margin
         if (AccountFreeMargin() < (1000 * Lots)) {
            Print("We have no money. Free Margin = ", AccountFreeMargin());
            return(0);
         }

         if (UseStopLoss) StopLossLevel = Bid + StopLoss * Point * P; else StopLossLevel = 0.0;
         if (UseTakeProfit) TakeProfitLevel = Bid - TakeProfit * Point * P; else TakeProfitLevel = 0.0;

         Ticket = OrderSend(Symbol(), OP_SELL, Lots, Bid, Slippage, StopLossLevel, TakeProfitLevel, "Sell(#" + MagicNumber + ")", MagicNumber, 0, DeepPink);
         if(Ticket > 0) {
            if (OrderSelect(Ticket, SELECT_BY_TICKET, MODE_TRADES)) {
				Print("SELL order opened : ", OrderOpenPrice());
                if (SignalMail) SendMail("[Signal Alert]", "[" + Symbol() + "] " + DoubleToStr(Bid, Digits) + " Open Sell");
			} else {
				Print("Error opening SELL order : ", GetLastError());
			}
         }
         return(0);
      }
   }

   return(0);
}
//+------------------------------------------------------------------+

bool New_Bar() {
   static datetime New_Time=0; // New_Time = 0 when New_Bar() is first called
   if(New_Time!=Time[0]){      // If New_Time is not the same as the time of the current bar's open, this is a new bar
      New_Time=Time[0];        // Assign New_Time as time of current bar's open
      return(true);
   }
   return(false);
}