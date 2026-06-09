import { useStore } from './store'
import Landing from './components/Landing'
import Instructions from './components/Instructions'
import DetectView from './components/DetectView'
import LocalizeView from './components/LocalizeView'
import Complete from './components/Complete'

export default function App() {
  const { screen, task } = useStore()

  switch (screen) {
    case 'landing':
      return <Landing />
    case 'instructions':
      return <Instructions />
    case 'annotating':
      return task === 'detect' ? <DetectView /> : <LocalizeView />
    case 'complete':
      return <Complete />
  }
}
