//+------------------------------------------------------------------+
//|                                                    OrderBook.mqh |
//|                                                    Jaydin Gulley |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict

class OrderBook {
private:
    string m_Data[];
    int m_Size;
    string m_Unique[];
    int m_UniqueSize;
    string m_NumUnique[];  // Change to string array for precision handling
    int m_NumUniqueSize;
    string m_Volume[];     // Change to string array for precision handling
    bool IsUnique(string Val);
    void UpdateNumUnique();

public:
    OrderBook();
    ~OrderBook();
    void Insert(string Data, long Volume_);
    void ClearBook();
    void dump();
    int SetArraysWithValues(string &UniqueValues[], int &UniqueNumValues[]);  // Change to string array
};

OrderBook::OrderBook(void) {
    ArrayResize(m_Data, 1);
    ArrayResize(m_Unique, 1);
    ArrayResize(m_NumUnique, 1); // Change to string array
    ArrayResize(m_Volume, 1);    // Change to string array
    m_Data[0] = "";
    m_Unique[0] = "";
    m_NumUnique[0] = "0";  // Initialize as string
    m_Volume[0] = "0";     // Initialize as string
    m_Size = 1;
    m_UniqueSize = 1;
    m_NumUniqueSize = 1;
}

OrderBook::~OrderBook(void) {}

void OrderBook::Insert(string Data, long Volume_) {
    string volumeStr = DoubleToString(Volume_, 10);  // Convert double to string with precision
    if (IsUnique(Data)) {
        m_Unique[m_UniqueSize - 1] = Data;
        m_UniqueSize++;
        ArrayResize(m_Unique, m_UniqueSize);
        ArrayResize(m_NumUnique, m_UniqueSize);
    }

    m_Data[m_Size - 1] = Data;
    m_Volume[m_Size - 1] = volumeStr;
    m_Size++;
    ArrayResize(m_Data, m_Size);
    ArrayResize(m_Volume, m_Size);
    UpdateNumUnique();
}

bool OrderBook::IsUnique(string Val) {
    for (int i = 0; i < m_UniqueSize - 1; i++) {
        if (Val == m_Unique[i]) {
            return false;
        }
    }
    return true;
}

void OrderBook::UpdateNumUnique(void) {
    for (int i = 0; i < m_UniqueSize - 1; i++) {
        double count = 0.0;
        for (int j = 0; j < m_Size - 1; j++) {
            if (m_Unique[i] == m_Data[j]) {
                count += StringToDouble(m_Volume[j]);  // Convert string back to double for calculation
            }
        }
        m_NumUnique[i] = DoubleToString(count, 10);  // Convert double to string
    }
}

int OrderBook::SetArraysWithValues(string &UniqueValues[], int &UniqueNumValues[]) {
    ArrayResize(UniqueValues, m_UniqueSize);
    ArrayResize(UniqueNumValues, m_UniqueSize);
    for (int i = 0; i < m_UniqueSize - 1; i++) {
        UniqueValues[i] = m_Unique[i];
        UniqueNumValues[i] = m_NumUnique[i];  // Return as string
    }
    return m_UniqueSize - 1;
}


void OrderBook::dump(void) {
    Print("m_Data:");
    for (int i = 0; i < m_Size - 1; i++) {
        Print(m_Data[i]);
    }

    Print("m_UniqueData:");
    for (int i = 0; i < m_UniqueSize - 1; i++) {
        Print(m_Unique[i], "---->", m_NumUnique[i]);  // Print as string
    }
}


void OrderBook::ClearBook(void) {
    ArrayResize(m_Data, 1);
    ArrayResize(m_Unique, 1);
    ArrayResize(m_NumUnique, 1);
    ArrayResize(m_Volume, 1);
    m_Data[0] = "";
    m_Unique[0] = "";
    m_NumUnique[0] = "0";  // Initialize as string
    m_Volume[0] = "0";     // Initialize as string
    m_Size = 1;
    m_UniqueSize = 1;
    m_NumUniqueSize = 1;
}
