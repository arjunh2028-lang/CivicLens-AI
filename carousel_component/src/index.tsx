import React from "react"
import { createRoot } from "react-dom/client"
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import { MinimalCarousel } from "./MinimalCarousel"
import "./index.css"

import { Activity, AlertTriangle, ShieldCheck, MapPin, Truck } from "lucide-react"

const ICONS = {
  Activity,
  AlertTriangle,
  ShieldCheck,
  MapPin,
  Truck,
}

class CarouselComponent extends StreamlitComponentBase {
  public render = (): React.ReactNode => {
    // Parse arguments from python
    const args = this.props.args
    const rawCards = args["cards"] || []
    
    const cards = rawCards.map((c: any) => ({
      ...c,
      icon: ICONS[c.icon as keyof typeof ICONS] || Activity
    }))

    return (
      <div className="w-full flex justify-center py-4">
        <MinimalCarousel 
          cards={cards} 
          onCopyClick={(card) => {
             Streamlit.setComponentValue({ action: "copy", cardId: card.id })
          }}
          onCustomizeClick={(card) => {
             Streamlit.setComponentValue({ action: "customize", cardId: card.id })
          }}
        />
      </div>
    )
  }
}

// Wrap and export
const StreamlitCarousel = withStreamlitConnection(CarouselComponent)

const root = createRoot(document.getElementById("root") as HTMLElement)
root.render(
  <React.StrictMode>
    <StreamlitCarousel />
  </React.StrictMode>
)
