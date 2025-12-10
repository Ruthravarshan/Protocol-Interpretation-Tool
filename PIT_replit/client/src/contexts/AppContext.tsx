import { createContext, useContext, useState, useEffect } from "react";

interface AppState {
  selectedExtractionId: string;
  setSelectedExtractionId: (id: string) => void;
  clearSelectedExtraction: () => void;
}

const AppContext = createContext<AppState | undefined>(undefined);

const STORAGE_KEY = "clinical_pdf_app_state";

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [selectedExtractionId, setSelectedExtractionIdState] = useState<string>(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        return parsed.selectedExtractionId || "";
      } catch {
        return "";
      }
    }
    return "";
  });

  const setSelectedExtractionId = (id: string) => {
    setSelectedExtractionIdState(id);
  };

  const clearSelectedExtraction = () => {
    setSelectedExtractionIdState("");
  };

  useEffect(() => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ selectedExtractionId })
    );
  }, [selectedExtractionId]);

  return (
    <AppContext.Provider
      value={{
        selectedExtractionId,
        setSelectedExtractionId,
        clearSelectedExtraction,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
}
